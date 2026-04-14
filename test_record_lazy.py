import json
import warnings
import os
import time

# 경고 메시지 숨기기 (JSON 출력 오염 방지)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow 로그 숨기기

import cv2
import mediapipe as mp
from pydub import AudioSegment
import whisper
from pydub.utils import which
import numpy as np
import math
import re

# =============================================================================
# 1. 초기 설정 및 모델 로드
# =============================================================================

# MediaPipe Face Mesh 설정 (얼굴 랜드마크 검출)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,       # 동영상 분석이므로 False (이전 프레임 정보 활용하여 추적 성능 향상)
    max_num_faces=1,               # 분석 대상은 1명으로 제한
    refine_landmarks=True,         # 눈동자 등 정교한 랜드마크 포함 (시선 분석에 필수)
    min_detection_confidence=0.5,  # 감지 신뢰도 임계값 (0.5 이상 추천)
    min_tracking_confidence=0.5    # 추적 신뢰도 임계값 (0.5 이상 추천)
)

# MediaPipe Pose 설정 (신체 동작 검출)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,            # 모델 복잡도 (0, 1, 2 중 선택, 1이 중간)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Whisper 모델 로드 (음성 인식)
# "base" 모델 사용 (속도와 정확도의 균형, 필요시 "small", "medium" 등으로 변경 가능)
# Lazy Loading 적용: 전역 로드 제거
model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model("base")
    return model

# =============================================================================
# 2. 오디오 처리 및 분석 함수
# =============================================================================
# ffmpeg/ffprobe 경로 설정 (pydub 라이브러리 사용을 위함)
def _ensure_ffmpeg_for_pydub():
    ffmpeg_path = which("ffmpeg")
    ffprobe_path = which("ffprobe")

    # 시스템 PATH에 없는 경우, 일반적인 설치 경로 확인 및 추가
    if not ffmpeg_path or not ffprobe_path:
        os.environ["PATH"] += os.pathsep + r"C:\\ProgramData\\chocolatey\\bin"
        ffmpeg_path = ffmpeg_path or which("ffmpeg")
        ffprobe_path = ffprobe_path or which("ffprobe")

    # 그래도 없으면 하드코딩된 경로 사용 (사용자 환경에 맞게 수정 필요)
    if not ffmpeg_path:
        ffmpeg_path = r"C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe"
    if not ffprobe_path:
        ffprobe_path = r"C:\\ProgramData\\chocolatey\\bin\\ffprobe.exe"

    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

def extract_audio(video_path):
    """영상 파일에서 오디오를 추출하여 wav 파일로 저장합니다."""
    _ensure_ffmpeg_for_pydub()
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio_path = "temp_audio.wav"
    audio.export(audio_path, format="wav")
    return audio_path

def whisper_transcribe(audio_path):
    """Whisper 모델을 사용하여 오디오를 텍스트로 변환합니다."""
    m = get_model()
    result = m.transcribe(
        audio_path,
        word_timestamps=True,
        initial_prompt="어, 음, 그, 저, 뭐, 아, 어어, 음음, 어... 음... 그..."
    )
    return result

def count_fillers_from_words(segs: list) -> int:
    """
    Whisper word_timestamps 기반 필러 감지.

    Whisper는 "어", "음" 등을 텍스트에서 제거하는 대신
    인접 단어의 duration 안으로 흡수시킴.
    예: "어.. 오늘은" → word "오늘은" start=0.0, end=2.4 (3글자인데 2.4초)

    감지 방식: 단어 지속시간 이상 감지
    - 한국어 정상 발화 속도: 글자당 약 0.15~0.25초
    - 글자당 duration이 DURATION_PER_CHAR_THRESHOLD 초 초과 시 필러 흡수로 판단
    - 초과된 시간 / 필러 1개당 평균 길이(0.5초)로 필러 개수 추정
    """
    DURATION_PER_CHAR_THRESHOLD = 0.35  # 글자당 이 초를 넘으면 비정상
    FILLER_AVG_SEC = 0.5                # 필러 1개당 평균 지속시간

    filler_cnt = 0

    for seg in segs:
        words = seg.get("words", [])
        if not words:
            continue

        for word_info in words:
            word = word_info.get("word", "").strip()
            duration = word_info.get("end", 0) - word_info.get("start", 0)
            char_count = max(1, len(word))
            duration_per_char = duration / char_count

            if duration_per_char > DURATION_PER_CHAR_THRESHOLD:
                # 정상 발화로 예상되는 시간 = char_count * 0.2초
                expected_duration = char_count * 0.2
                excess = duration - expected_duration
                # 초과 시간을 필러 평균 길이로 나눠 개수 추정
                estimated = max(1, round(excess / FILLER_AVG_SEC))
                filler_cnt += estimated
                print(f"[Filler] '{word}' duration={duration:.2f}s ({duration_per_char:.2f}s/char) → +{estimated} filler")

    print(f"[Filler] total filler_cnt={filler_cnt}")
    return filler_cnt


def speech_stats(transcribe_result, min_seconds=3.0):
    """
    STT 결과에서 발화 속도(WPM)와 필러(추임새) 사용 빈도를 계산합니다.
    필러는 word_timestamps 기반 pause 감지로 측정 (Whisper가 텍스트에서 제거하므로).

    Returns:
        dict: {text, wpm, cps, duration, fillers_count, fillers_freq,
               silence_count, total_silence_sec, silence_ratio}
    """
    segs = transcribe_result.get("segments", [])
    text = transcribe_result.get("text", "").strip()

    if not segs:
        return {
            "text": text, "wpm": 0.0, "cps": 0.0, "duration": 0.0,
            "fillers_count": 0, "fillers_freq": 0.0,
            "silence_count": 0, "total_silence_sec": 0.0, "silence_ratio": 0.0,
        }

    start = segs[0]["start"]
    end = segs[-1]["end"]
    dur = max(0.0, end - start)

    # 단어 수 및 글자 수
    words = sum(len(s["text"].split()) for s in segs)
    chars = sum(len(s["text"].replace(" ", "")) for s in segs)

    wpm = (words / (dur / 60.0)) if dur >= min_seconds else 0.0
    cps = (chars / dur) if dur > 0 else 0.0

    # 필러 감지 (두 방식 병행)
    # 1차: initial_prompt 덕에 텍스트에 직접 나온 필러 카운팅 (종류 구분 가능)
    FILLERS = {
        "어": "hesitation",
        "음": "hesitation",
        "그": "filler",
        "저": "filler",
        "뭐": "filler",
        "아": "hesitation",
        "um": "hesitation",
        "uh": "hesitation",
    }
    text_filler_counts = {label: 0 for label in set(FILLERS.values())}
    text_filler_counts["total"] = 0
    filler_list = []  # 등장 순서대로 실제 필러 단어 목록
    for word, label in FILLERS.items():
        pattern = r"(?<!\w)" + re.escape(word) + r"(?!\w)"
        matches = re.findall(pattern, text)
        cnt = len(matches)
        text_filler_counts[label] += cnt
        text_filler_counts["total"] += cnt
        filler_list.extend(matches)

    # 등장 순서 복원: text에서 순서대로 재추출
    ordered_filler_list = []
    filler_pattern = r"(?<!\w)(" + "|".join(re.escape(w) for w in FILLERS) + r")(?!\w)"
    for match in re.finditer(filler_pattern, text):
        ordered_filler_list.append(match.group(1))

    # 2차: duration 이상 감지 (텍스트 필러가 0이면 fallback)
    duration_filler_cnt = count_fillers_from_words(segs)

    # 텍스트 기반이 감지되면 우선 사용, 없으면 duration 기반 사용
    if text_filler_counts["total"] > 0:
        filler_cnt = text_filler_counts["total"]
        filler_detail = text_filler_counts
        print(f"[Filler] 텍스트 기반 감지: {filler_detail}, list={ordered_filler_list}")
    else:
        filler_cnt = duration_filler_cnt
        filler_detail = {"total": filler_cnt, "hesitation": 0, "filler": 0}
        ordered_filler_list = []
        print(f"[Filler] duration 기반 감지 fallback: {filler_cnt}")

    filler_freq = (filler_cnt / (dur / 60.0)) if dur >= min_seconds else 0.0

    # 침묵: 세그먼트 사이 gap >= 2초
    SILENCE_GAP_SEC = 2.0
    silence_gaps = []
    for i in range(1, len(segs)):
        gap = segs[i]["start"] - segs[i - 1]["end"]
        if gap >= SILENCE_GAP_SEC:
            silence_gaps.append(gap)

    silence_count = len(silence_gaps)
    total_silence_sec = round(sum(silence_gaps), 2)
    silence_ratio = round(total_silence_sec / dur, 3) if dur > 0 else 0.0

    return {
        "text": text,
        "wpm": float(wpm),
        "cps": float(cps),
        "duration": float(dur),
        "fillers_count": int(filler_cnt),
        "fillers_freq": float(round(filler_freq, 2)),
        "filler_detail": filler_detail,
        "filler_list": ordered_filler_list,  # ex. ["어", "음", "그", "어"]
        "silence_count": silence_count,
        "total_silence_sec": total_silence_sec,
        "silence_ratio": silence_ratio,
    }

# =============================================================================
# 3. 영상 처리 및 분석 함수 (시선, 포즈)
# =============================================================================

# =============================================================================
# 3. 영상 처리 및 분석 함수 (시선, 포즈)
# =============================================================================

# 공통 유틸리티에서 로직 가져오기 (코드 중복 제거)
from app.utils.analysis_utils import (
    gaze_from_landmarks, 
    movement_speed, 
    HAND_KEYS
)

# =============================================================================
# 4. 메인 분석 파이프라인
# =============================================================================

def analyze_video(video_path):
    start_time = time.time()
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(json.dumps({"error": "비디오 파일을 열 수 없습니다."}, ensure_ascii=False))
        return
    
    frame_count = 0
    
    # 분석 설정
    FRAME_STRIDE = 5               # 5프레임마다 1번 분석 (속도 최적화)
    speeds = []                    # 손/팔 움직임 속도 기록 리스트
    prev_pose_points = None
    gaze_samples = []              # 시선 데이터 기록 리스트
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 설정한 간격대로 프레임 건너뛰기
        if frame_count % FRAME_STRIDE != 0:
            continue
        
        # 처리 속도를 위해 리사이즈
        frame = cv2.resize(frame, (640, 360))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. FaceMesh 처리 (시선 분석)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            gaze = gaze_from_landmarks(lms)
            gaze_samples.append(gaze)
            
        # 2. Pose 처리 (손/팔 움직임 분석)    
        p_res = pose.process(rgb)
        curr_points = None
        if p_res.pose_landmarks:
            curr_points = {i: (lm.x, lm.y) for i, lm in enumerate(p_res.pose_landmarks.landmark)}
        
        spd = movement_speed(prev_pose_points, curr_points)
        if spd is not None:
            speeds.append(spd)
        prev_pose_points = curr_points

    video.release()
    
    # 3. 오디오 분석 (말하기 속도, 필러)
    audio_path = extract_audio(video_path)
    stt_result = whisper_transcribe(audio_path)
    speech = speech_stats(stt_result)
    
    # 4. 결과 집계 및 JSON 생성
    
    # 손/팔 움직임 평균 속도
    avg_speed = float(np.mean(speeds)) if speeds else 0.0

    # 시선 데이터 요약
    if gaze_samples:
        horiz_counts = {"left":0, "center":0, "right":0}
        vert_counts  = {"up":0, "center":0, "down":0}
        for g in gaze_samples:
            horiz_counts[g["horiz"]] += 1
            vert_counts[g["vert"]]   += 1
            
        horiz_mode = max(horiz_counts, key=horiz_counts.get)
        vert_mode  = max(vert_counts,  key=vert_counts.get)
        avg_dx = float(np.mean([g["dx"] for g in gaze_samples]))
        avg_dy = float(np.mean([g["dy"] for g in gaze_samples]))
        
        eyes_summary = {
            "avg_dx": round(avg_dx, 4),      # 좌우 평균 편차
            "avg_dy": round(avg_dy, 4),      # 상하 평균 편차
            "horiz_mode": horiz_mode,        # 주로 본 좌우 방향
            "vert_mode": vert_mode,          # 주로 본 상하 방향
            "samples": len(gaze_samples)     # 분석된 프레임 수
        }
    else:
        eyes_summary = {
            "avg_dx": 0.0, "avg_dy": 0.0, 
            "horiz_mode": "n/a", "vert_mode": "n/a", 
            "samples": 0
        }

    # 최종 결과 딕셔너리 생성
    final_result = {
        "WPM": round(speech["wpm"], 1),                # 말하기 속도 (Words Per Minute)
        "filer": round(speech["fillers_freq"], 1),     # 필러 사용 빈도 (회/분)
        "handArmMovementAvg": round(avg_speed, 5),     # 손/팔 움직임 평균 속도 (정규화 거리 단위)
        "eyes": eyes_summary                           # 시선 분석 결과
    }

    elapsed = time.time() - start_time
    print(f"[Sync] 완료. 소요시간: {elapsed:.2f}s")

    # JSON 형태로 출력
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
    
    # (선택사항) 상세 텍스트 저장
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(speech["text"])
        
    return final_result

if __name__ == "__main__":
    # 스크립트 실행 시 video.mp4 분석
    analyze_video("video_standard.mp4")
