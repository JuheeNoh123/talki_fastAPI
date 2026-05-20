# app/services/speech_service.py
# 음성 분석 서비스 (완전 독립 모듈 - cv2/mediapipe 의존 없음)
# 기능:
#   - analyze_chunk()        : 실시간 청크 말속도 분석
#   - analyze_full_audio()   : 전체 오디오 말속도 최종 분석
#   - analyze_pronunciation(): 전체 오디오 발음 정확도 분석

from __future__ import annotations

import multiprocessing
import tempfile, wave, os
import numpy as np
import torch

from app.config.feedback_criteria import FEEDBACK_CRITERIA


# =============================================================================
# Whisper worker 함수 (별도 프로세스에서 실행)
# =============================================================================
def _whisper_worker(conn):
    """Whisper 모델을 로드하고 Pipe로 요청을 받아 STT 처리"""
    try:
        print("[SpeechWhisper] 초기화 중...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SpeechWhisper] Device: {device}")

        import whisper
        model = whisper.load_model("small", device=device)
        print("[SpeechWhisper] 모델 로드 완료. 대기 중...")

        while True:
            if not conn.poll(timeout=None):
                continue

            audio_path = conn.recv()
            if audio_path is None:  # 종료 신호
                break

            initial_prompt = "어, 음, 그, 저, 뭐, 아"

            print(f"[SpeechWhisper] STT 요청 수신: {audio_path}")

            try:
                result = model.transcribe(
                    audio_path,
                    language="ko",
                    word_timestamps=True,
                    initial_prompt=initial_prompt
                )
                stats = _calc_wpm(result)
                conn.send({"status": "success", "data": stats})
                print(f"[SpeechWhisper] 완료 | WPM: {stats['wpm']:.1f}")

            except Exception as e:
                print(f"[SpeechWhisper] 에러: {e}")
                conn.send({"status": "error", "message": str(e)})

    except Exception as e:
        try:
            conn.send({"status": "fatal_error", "message": str(e)})
        except:
            pass
    finally:
        print("[SpeechWhisper] 프로세스 종료")


# 별도 worker 함수를 사용하는 이유.
# 기존 whisper_service(녹화 영상 분석용)를 재사용하지 않고 자체 worker를 두는 이유는 사용 목적이 달라서.

# whisper_service는 영상 분석 파이프라인(test_record_multiprocess)의 일부로,
# import 시점에 MediaPipe(FaceMesh, Pose) 모델까지 함께 초기화됨.
# 하지만 이 서비스에서는 순수 음성 분석(말속도, 발음 평가)만 필요하므로
# cv2/mediapipe 의존성이 없는 경량 worker를 별도로 운영함.

# 실제로 기존 whisper_service를 공유하려 했을 때 MediaPipe 초기화가
# 메인 프로세스 import 단계에서 freeze를 일으켜 정상 동작하지 않았음.

# A의 방식으로 수정하는 방향이 맞다고 생각이 된다면, speech_service에서 선언한 whisper을 whisper_service에서 가져다쓰는 방향이 더 알맞지 않을까 싶음


# =============================================================================
# WPM 계산 함수
# =============================================================================
def _calc_wpm(transcribe_result, min_seconds=3.0) -> dict:
    """Whisper 결과에서 WPM 계산"""
    segs = transcribe_result.get("segments", [])
    text = transcribe_result.get("text", "").strip()

    if not segs:
        return {"text": text, "wpm": 0.0}

    # 세그먼트 사이 침묵 gap을 제외한 실제 발화 시간만 합산
    speaking_dur = sum(s["end"] - s["start"] for s in segs)

    # 앞뒤 공백 제거 후 split — 공백만 있는 세그먼트는 단어 수 0으로 처리
    words = sum(len(s["text"].strip().split()) for s in segs if s["text"].strip())
    wpm   = (words / (speaking_dur / 60.0)) if speaking_dur >= min_seconds else 0.0

    return {"text": text, "wpm": round(wpm, 1)}


# =============================================================================
# Whisper 프로세스 관리 클래스
# =============================================================================
class _SpeechWhisperService:
    # 메인 프로세스와 whisper 프로세스 사이 통신 채널과 프로세스 객체 생성
    def __init__(self):
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.process = multiprocessing.Process(
            target=_whisper_worker,
            args=(self.child_conn,),
            daemon=True
        )
        self.started = False

    # whisper worker 프로세스 실제로 띄움. started로 중복 실행 방지
    def start(self):
        if not self.started:
            self.process.start()
            self.started = True

    # 오디오 파일 경로를 pipe로 worker에게 전달. 결과 안 기다리고 반환.
    def transcribe_async(self, audio_path: str):
        self.parent_conn.send(audio_path)

    # worker가 처리 완료한 결과를 pipe에서 꺼냄. 결과 올 때까지 블로킹.
    def get_result(self) -> dict:
        return self.parent_conn.recv()


# Whisper 인스턴스 (첫 호출 시 초기화)
_whisper: _SpeechWhisperService = None

def _get_whisper() -> _SpeechWhisperService:
    """Lazy initialization - 처음 호출 시에만 프로세스 시작"""
    global _whisper
    if _whisper is None:
        _whisper = _SpeechWhisperService()
        _whisper.start()
    return _whisper


# =============================================================================
# 공통 헬퍼: numpy 배열 → wav 저장 후 Whisper 실행 → 결과 반환
# =============================================================================
def _transcribe(audio_np: np.ndarray) -> dict | None:
    """
    세 분석 함수(analyze_chunk, analyze_full_audio, analyze_pronunciation)의
    공통 로직을 처리한다.
    - numpy 배열을 임시 wav 파일로 저장
    - Whisper worker에 전달 후 결과 수신
    - 임시 파일 삭제
    실패 시 None 반환
    """
    # 임시 wav 파일 생성 (delete=False: Whisper가 나중에 열 수 있도록 with 블록 후에도 유지)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

    # numpy 배열을 Whisper가 읽을 수 있는 wav 형식으로 저장
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)     # 모노(음성 분석에 방향x) : 1 / 스테레오(음성 방향 포함) : 2 => 방향이 불필요하기 때문에 1
        wf.setsampwidth(2)     # 16bit — audio_np가 int16이므로 1샘플 = 2바이트
        wf.setframerate(16000) # Whisper 요구 샘플레이트 — 프론트에서 16000Hz로 맞춰 전송 => 문제 X
        wf.writeframes(audio_np.tobytes())

    # Whisper에 오디오를 넘기고 결과를 받아오는 코드. 
    try:
        whisper = _get_whisper()
        whisper.transcribe_async(wav_path)
        result = whisper.get_result()

        if result["status"] != "success":
            print(f"[SpeechService] STT 실패: {result.get('message')}")
            return None

        return result["data"]

    # 예외 발생 시 None 반환. finally - 성공/실패 관계없이 임시 파일을 항상 삭제
    except Exception as e:
        print(f"[SpeechService] 에러: {e}")

        #에러 발생했을때 어디 코드 몇번째 줄에서 에러가 났는지 출력해주는 디버깅용 코드
        import traceback
        traceback.print_exc()

        return None

    finally:
        os.remove(wav_path)


# =============================================================================
# 1단계: 실시간 청크 말속도 분석
# =============================================================================
def analyze_chunk(audio_np: np.ndarray, presentation_type: str = "small") -> dict:
    """
    5초 청크 오디오를 받아서 실시간 WPM + 속도 상태 반환

    Args:
        audio_np: 오디오 numpy 배열 (int16, 16000Hz, mono)
        presentation_type: 발표 유형 (small / large / online_small)

    Returns:
        {"wpm": float, "speed_status": "fast" | "slow" | "normal"}
        실패 시 None 반환
    """
    # 발표 유형별 WPM 기준 가지고 옴. 기본값은 발표 유형(small)
    criteria = FEEDBACK_CRITERIA.get(presentation_type, FEEDBACK_CRITERIA["small"])

    # 공통 헬퍼로 음성 파일 저장 -> whisper 실행 -> 결과 수신 처리. 
    data = _transcribe(audio_np)
    if data is None:
        return None

    # 헬퍼에서 반환한 결과에서 WPM 수치만 꺼냄
    wpm = data["wpm"]

    # 꺼낸 수치를 기준값과 비교해 속도 상태를 판정. 
    if wpm > criteria["wpm_max"]:
        speed_status = "fast"
    elif 0 < wpm < criteria["wpm_min"]:
        speed_status = "slow"
    else:
        speed_status = "normal"

    # 결과 반환
    print(f"[SpeechService] WPM: {wpm} → {speed_status}")
    return {"wpm": wpm, "speed_status": speed_status}


# =============================================================================
# WPM 점수 계산 (feedback_service import 없이 독립적으로 사용)
# 80~110 WPM이면 100점, 벗어날수록 선형 감점
# =============================================================================
def _calc_wpm_score(wpm: float, criteria: dict) -> int:
    min_wpm = criteria["wpm_min"]
    max_wpm = criteria["wpm_max"]
    if wpm == 0:
        return 0
    if min_wpm <= wpm <= max_wpm:
        return 100
    diff = (min_wpm - wpm) / min_wpm if wpm < min_wpm else (wpm - max_wpm) / max_wpm
    return int(max(0, min(100, 100 - diff * 100)))


# =============================================================================
# 2단계: 전체 오디오 말속도 최종 분석
# 위 코드(실시간)과 큰 차이는 없음. 라우터에서 그때마다 필요한 함수를 호출하는 방향이라 
# 결과 출력의 차이점이 있음.
# =============================================================================
def analyze_full_audio(audio_np: np.ndarray, presentation_type: str = "small") -> dict:
    """
    세션 종료 후 누적된 전체 오디오로 최종 WPM + 점수 계산

    Args:
        audio_np: 전체 누적 오디오 numpy 배열 (int16, 16000Hz, mono)
        presentation_type: 발표 유형 (small / large / online_small)

    Returns:
        {"wpm": float, "wpm_score": int}
        실패 시 None 반환
    """
    criteria = FEEDBACK_CRITERIA.get(presentation_type, FEEDBACK_CRITERIA["small"])

    data = _transcribe(audio_np)
    if data is None:
        return None

    wpm = data["wpm"]
    wpm_score = _calc_wpm_score(wpm, criteria)

    print(f"[SpeechService] 최종 WPM: {wpm} | 점수: {wpm_score}")
    return {"wpm": wpm, "wpm_score": wpm_score}


# =============================================================================
# 3단계 헬퍼: 한국어 자모 분해
# =============================================================================
_CHOSUNG  = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNGSUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONGSUNG = list(" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

# 한국어 자모 분해. 
def _decompose_jamo(text: str) -> str:
    result = []
    for char in text:
        code = ord(char)
        # 유니코드 수식. 한글(가~힣) 0xAC00부터 순서대로 배치되어 있고
        # 한글자의 코드값에서 0xAC00을 빼면 다음 공식으로 자모를 역산할 수 있음
        # 종성 index = code % 28
        # 중성 index = (code // 28) % 21
        # 초성 index = code // (28 * 21)

        # => 이걸 하는 이유는 글자단위로 비교하면 '발'과 '바'차이가 1인데, 
        # 자모 단위면 'ㄹ' 하나 차이라는 걸 알 수 있음.
        if 0xAC00 <= code <= 0xD7A3:
            code -= 0xAC00
            jong = code % 28 # 종성
            code //= 28
            jung = code % 21 # 중성
            cho  = code // 21 # 초성
            result.append(_CHOSUNG[cho])
            result.append(_JUNGSUNG[jung])
            if jong != 0:
                result.append(_JONGSUNG[jong])
        else:
            result.append(char)
    return "".join(result)


def _edit_distance(s1: str, s2: str) -> int:
    """레벤슈타인 거리 계산 (자모열 비교에 사용)"""
    # 두 문자열이 얼마나 다른지 숫자로 계산. 
    # (ex) 'ㅂㅏㄹㅇㅡㅁ' -> 'ㅂㅏㅇㅡㅁ' : 거리 1
    # DP 방식으로 구현
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    # 이걸 통해서 발표와 바표를 똑같은 2글자로 인식하지 않고 종성이 빠진걸 잡을 수 있음
    # 자모로 풀어놓은 문자열을 DP로 비교.
    return dp[n]


# 자모 분리하고, 빠진 걸 찾았으니, 이젠 틀린 거 찾기.
def _find_errors(reference: str, recognized: str) -> list:
    import re, difflib

    ref_words = re.sub(r'[^\w가-힣\s]', '', reference).split()
    rec_words = re.sub(r'[^\w가-힣\s]', '', recognized).split()

    # difflib.SequenceMatcher 두 단어 리스트의 유사도 비교.
    # 이게 두 단어 리스트의 공통 부분을 찾고 나머지를 replace/delete/insert 태그로 분류
    matcher = difflib.SequenceMatcher(None, ref_words, rec_words)
    errors = []

    # 기존 텍스트와 인식된 텍스트를 단어 단위로 비교.
    # 어느 부분이 틀렸는지 목록으로 반환.
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # 단어가 바뀜
        if tag == "replace":
            errors.append({
                "reference":  " ".join(ref_words[i1:i2]),
                "recognized": " ".join(rec_words[j1:j2])
            })
        # 기존엔 있는데 안 말함
        elif tag == "delete":
            errors.append({
                "reference":  " ".join(ref_words[i1:i2]),
                "recognized": "(없음)"
            })
        # 안 해야할 말을 함
        elif tag == "insert":
            errors.append({
                "reference":  "(없음)",
                "recognized": " ".join(rec_words[j1:j2])
            })

    return errors


# =============================================================================
# 3단계: 전체 오디오 발음 정확도 분석
# =============================================================================
def analyze_pronunciation(audio_np: np.ndarray, reference_text: str) -> dict:
    """
    전체 오디오와 기준 텍스트를 비교해서 발음 정확도 점수 반환

    Args:
        audio_np: 전체 누적 오디오 numpy 배열 (int16, 16000Hz, mono)
        reference_text: 사용자가 읽어야 할 기준 텍스트

    Returns:
        {
            "pronunciation_score": float,   # 0~100점
            "cer": float,                   # 자모 오류율 (낮을수록 좋음)
            "recognized_text": str,         # Whisper가 인식한 텍스트
            "errors": list                  # 틀린 단어 목록
        }
        실패 시 None 반환
    """
    import re

    # Whisper로 오디오를 텍스트로 변환
    data = _transcribe(audio_np)
    if data is None:
        return None

    recognized_text = data["text"]
    print(f"[SpeechService] 인식된 텍스트: {recognized_text}")

    # 특수문자 제거
    ref_clean = re.sub(r'[^\w가-힣]', '', reference_text)
    rec_clean = re.sub(r'[^\w가-힣]', '', recognized_text)

    # 자모분해 -> 편집 거리 -> CER
    ref_jamo = _decompose_jamo(ref_clean)
    rec_jamo = _decompose_jamo(rec_clean)

    dist = _edit_distance(ref_jamo, rec_jamo)
    cer  = round(min(dist / len(ref_jamo), 1.0), 4) if ref_jamo else 1.0

    # 패널티 3.5를 곱해서 점수 감점을 증폭.
    # CER 기준에 맞춰 패널티를 지정. 
    # 현재는 
    # CER 1.65% => 94점
    # CER 16.54% => 42점
    # 이 나오는 3.5 패널티로 맞춰둔 상태.
    PENALTY = 3.5
    pronunciation_score = round(max(0.0, (1.0 - cer * PENALTY) * 100), 1)

    # 틀린 단어 목록
    errors = _find_errors(reference_text, recognized_text)

    print(f"[SpeechService] CER: {cer} | 발음 점수: {pronunciation_score}")
    print(f"[SpeechService] 틀린 부분: {errors if errors else '없음'}")

    return {
        "pronunciation_score": pronunciation_score,
        "cer": cer,
        "recognized_text": recognized_text,
        "errors": errors,
    }
