# app/services/speech_service.py
# 음성 분석 서비스 (완전 독립 모듈 - cv2/mediapipe 의존 없음)
# 기능:
#   - analyze_chunk()        : 실시간 청크 말속도 분석
#   - analyze_full_audio()   : 전체 오디오 말속도 최종 분석
#   - analyze_pronunciation(): 전체 오디오 발음 정확도 분석

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


# =============================================================================
# WPM 계산 함수
# =============================================================================
def _calc_wpm(transcribe_result, min_seconds=3.0) -> dict:
    """Whisper 결과에서 WPM 계산"""
    segs = transcribe_result.get("segments", [])
    text = transcribe_result.get("text", "").strip()

    if not segs:
        return {"text": text, "wpm": 0.0}

    start = segs[0]["start"]
    end   = segs[-1]["end"]
    dur   = max(0.0, end - start)

    words = sum(len(s["text"].split()) for s in segs)
    wpm   = (words / (dur / 60.0)) if dur >= min_seconds else 0.0

    return {"text": text, "wpm": round(wpm, 1)}


# =============================================================================
# Whisper 서비스 (독립 프로세스 관리)
# =============================================================================
class _SpeechWhisperService:
    def __init__(self):
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.process = multiprocessing.Process(
            target=_whisper_worker,
            args=(self.child_conn,),
            daemon=True
        )
        self.started = False

    def start(self):
        if not self.started:
            self.process.start()
            self.started = True

    def transcribe_async(self, audio_path: str):
        self.parent_conn.send(audio_path)

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
    criteria = FEEDBACK_CRITERIA.get(presentation_type, FEEDBACK_CRITERIA["small"])
    print(f"[SpeechService] 청크 분석 | 길이: {len(audio_np)/16000:.1f}초 | 기준: {criteria['wpm_min']}~{criteria['wpm_max']} WPM")

    # 임시 wav 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_np.tobytes())

    try:
        whisper = _get_whisper()
        whisper.transcribe_async(wav_path)
        result = whisper.get_result()
        print(f"[SpeechService] Whisper 응답: {result.get('status')}")

        if result["status"] != "success":
            print(f"[SpeechService] STT 실패: {result.get('message')}")
            return None

        wpm = result["data"]["wpm"]

        if wpm > criteria["wpm_max"]:
            speed_status = "fast"
        elif 0 < wpm < criteria["wpm_min"]:
            speed_status = "slow"
        else:
            speed_status = "normal"

        print(f"[SpeechService] WPM: {wpm} → {speed_status}")
        return {"wpm": wpm, "speed_status": speed_status}

    except Exception as e:
        print(f"[SpeechService] 에러: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        os.remove(wav_path)


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
    print(f"[SpeechService] 전체 오디오 분석 | 길이: {len(audio_np)/16000:.1f}초")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_np.tobytes())

    try:
        whisper = _get_whisper()
        whisper.transcribe_async(wav_path)
        result = whisper.get_result()
        print(f"[SpeechService] Whisper 응답: {result.get('status')}")

        if result["status"] != "success":
            print(f"[SpeechService] STT 실패: {result.get('message')}")
            return None

        wpm = result["data"]["wpm"]
        wpm_score = _calc_wpm_score(wpm, criteria)

        print(f"[SpeechService] 최종 WPM: {wpm} | 점수: {wpm_score}")
        return {
            "wpm": wpm,
            "wpm_score": wpm_score,
        }

    except Exception as e:
        print(f"[SpeechService] 에러: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        os.remove(wav_path)


# =============================================================================
# 3단계 헬퍼: 한국어 자모 분해
# 예) "안녕" → "ㅇㅏㄴㄴㅕㅇ"
# 자모 단위로 비교하면 "서 vs 성" 같은 부분 발음 오류도 반영됨
# =============================================================================
_CHOSUNG  = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNGSUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONGSUNG = list(" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

def _decompose_jamo(text: str) -> str:
    result = []
    for char in text:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:  # 한글 음절 범위
            code -= 0xAC00
            jong = code % 28
            code //= 28
            jung = code % 21
            cho  = code // 21
            result.append(_CHOSUNG[cho])
            result.append(_JUNGSUNG[jung])
            if jong != 0:
                result.append(_JONGSUNG[jong])
        else:
            result.append(char)
    return "".join(result)


def _edit_distance(s1: str, s2: str) -> int:
    """레벤슈타인 거리 계산 (자모열 비교에 사용)"""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def _find_errors(reference: str, recognized: str) -> list:
    """
    기준 텍스트와 인식 텍스트를 단어 단위로 비교해서 다른 부분 반환

    Returns:
        [{"reference": "변화를", "recognized": "변화가"}, ...]
    """
    import re, difflib

    # 특수문자 제거 후 단어 분리
    ref_words = re.sub(r'[^\w가-힣\s]', '', reference).split()
    rec_words = re.sub(r'[^\w가-힣\s]', '', recognized).split()

    matcher = difflib.SequenceMatcher(None, ref_words, rec_words)
    errors = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            errors.append({
                "reference":  " ".join(ref_words[i1:i2]),
                "recognized": " ".join(rec_words[j1:j2])
            })
        elif tag == "delete":  # 기준엔 있는데 발화 안 됨
            errors.append({
                "reference":  " ".join(ref_words[i1:i2]),
                "recognized": "(없음)"
            })
        elif tag == "insert":  # 기준엔 없는데 발화됨
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

    print(f"[SpeechService] 발음 분석 시작 | 오디오 길이: {len(audio_np)/16000:.1f}초")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_np.tobytes())

    try:
        # initial_prompt에 reference_text를 넣으면 Whisper가 오발음도 기준 텍스트로 교정해버림
        # 발음 평가에서는 실제 들린 그대로 인식해야 하므로 기본 filler 프롬프트만 사용
        whisper = _get_whisper()
        whisper.transcribe_async(wav_path)
        result = whisper.get_result()

        if result["status"] != "success":
            print(f"[SpeechService] STT 실패: {result.get('message')}")
            return None

        recognized_text = result["data"]["text"]
        print(f"[SpeechService] 인식된 텍스트: {recognized_text}")  # 테스트용 출력

        # 특수문자 제거 후 자모 분해
        ref_clean = re.sub(r'[^\w가-힣]', '', reference_text)
        rec_clean = re.sub(r'[^\w가-힣]', '', recognized_text)

        ref_jamo = _decompose_jamo(ref_clean)
        rec_jamo = _decompose_jamo(rec_clean)

        # CER 계산
        dist = _edit_distance(ref_jamo, rec_jamo)
        cer  = round(min(dist / len(ref_jamo), 1.0), 4) if ref_jamo else 1.0

        # 점수 계산
        # NOISE_FLOOR: 이 이하의 CER은 완벽으로 간주 → 100점 가능
        # PENALTY: 노이즈 플로어 초과분에 대한 감점 강도 (실험으로 조정)
        PENALTY = 3.5
        pronunciation_score = round(max(0.0, (1.0 - cer * PENALTY) * 100), 1)

        # 틀린 단어 탐지
        errors = _find_errors(reference_text, recognized_text)

        print(f"[SpeechService] CER: {cer} | 발음 점수: {pronunciation_score}")
        print(f"[SpeechService] 틀린 부분: {errors if errors else '없음'}")

        return {
            "pronunciation_score": pronunciation_score,
            "cer": cer,
            "recognized_text": recognized_text,
            "errors": errors,
        }

    except Exception as e:
        print(f"[SpeechService] 에러: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        os.remove(wav_path)
