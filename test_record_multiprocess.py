import cv2
import numpy as np
import json
import time
import multiprocessing
import torch
import os
import threading
from test_record_lazy import (
    gaze_from_landmarks, 
    movement_speed, 
    extract_audio, 
    speech_stats
)

# =============================================================================
# 1. Whisper Worker Process (Persistent with Pipe)
# =============================================================================

def whisper_worker(conn):
    """
    Whisper 모델을 로드하고 요청을 처리하는 상주 프로세스 함수 (Pipe 사용)
    """
    try:
        print("[Whisper Process] 초기화 시작...")
        init_start = time.time()
        
        # GPU 확인 및 디바이스 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Whisper Process] Device: {device}")
        
        import whisper
        # 모델 로드 (최초 1회 실행)
        model = whisper.load_model("small", device=device)
        init_elapsed = time.time() - init_start
        print(f"[Whisper Process] 모델 로드 완료 (소요시간: {init_elapsed:.2f}s). 대기 중...")
        
        while True:
            # Pipe에서 작업 가져오기
            if not conn.poll(timeout=None): # 대기
                continue
                
            task = conn.recv()
            if task is None: # 종료 신호
                break
            
            # task는 audio_path
            audio_path = task
            print(f"[Whisper Process] STT 분석 요청 수신: {audio_path}")
            
            # 절대 시간 기록 (Overlap 계산용)
            abs_start = time.time()
            
            try:
                # Transcribe
                # initial_prompt: Whisper가 "어", "음" 등 필러를 suppression하지 않고
                # 실제 전사하도록 유도. word_timestamps는 타임스탬프 확보용.
                result = model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    initial_prompt="어, 음, 그, 저, 뭐, 아, 어어, 음음, 그래서, 근데, 어... 음... 그..."
                )
                
                # 통계 계산
                stats = speech_stats(result)
                
                abs_end = time.time()
                transcribe_elapsed = abs_end - abs_start
                print(f"[Whisper Process] 분석 완료 ({transcribe_elapsed:.2f}s)")
                
                # 결과 전송
                conn.send({
                    "status": "success", 
                    "data": stats, 
                    "timing": {
                        "init": init_elapsed,
                        "transcribe": transcribe_elapsed,
                        "abs_start": abs_start,
                        "abs_end": abs_end
                    }
                })
            except Exception as e:
                print(f"[Whisper Process] 분석 중 에러: {e}")
                conn.send({"status": "error", "message": str(e)})
            
    except Exception as e:
        try:
            conn.send({"status": "fatal_error", "message": str(e)})
        except:
            pass
    finally:
        print("[Whisper Process] 종료")

# Whisper를 독립된 프로세스로 실행
class WhisperService:
    def __init__(self):
        # Queue 대신 메인 프로세스와 통신하기 위한 Pipe 사용 (양방향)
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        # Whisper 작업을 전담할 별도 프로세스 생성
        self.process = multiprocessing.Process(
            target=whisper_worker, 
            args=(self.child_conn,),
            daemon=True
        )
        self.started = False

    def start(self):
        if not self.started:
            self.process.start()
            self.started = True

    def stop(self):
        if self.started:
            self.parent_conn.send(None)
            self.process.join()
            self.started = False

    def transcribe_async(self, audio_path):
        """비동기 요청 전송 (결과는 나중에 받음)"""
        self.parent_conn.send(audio_path)

    def get_result(self):
        """결과 수신 대기"""
        return self.parent_conn.recv()

# =============================================================================
# 2. Video Analysis Worker (Pool)
# =============================================================================

# 전역 변수로 Worker 프로세스 내의 모델 인스턴스 저장
worker_face_mesh = None
worker_pose = None

def init_worker():
    """Worker 프로세스 초기화 시 실행: 각 프로세스마다 별도의 MediaPipe 인스턴스 생성"""
    global worker_face_mesh, worker_pose
    import mediapipe as mp
    
    mp_face_mesh = mp.solutions.face_mesh
    worker_face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,  # 병렬/Stride 처리 시 프레임 연속성이 없으므로 True 필수
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    mp_pose = mp.solutions.pose
    worker_pose = mp_pose.Pose(
        static_image_mode=True, # 연속성이 깨지므로 True로 설정하여 독립적 감지 수행
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def process_frame_mp(frame):
    """Worker 프로세스에서 실행될 함수"""
    # 이미지 전처리
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 전역 객체 사용
    f_res = worker_face_mesh.process(rgb)
    p_res = worker_pose.process(rgb)
    
    # 결과 데이터 추출 (Pickle 가능한 데이터만 반환해야 함)
    gaze_data = None
    if f_res.multi_face_landmarks:
        lms = f_res.multi_face_landmarks[0].landmark
        gaze_data = gaze_from_landmarks(lms)
        
    pose_points = None
    if p_res.pose_landmarks:
        # visibility < 0.5인 랜드마크는 화면 밖 추정값 → 제외
        # HAND_KEYS(손목/팔꿈치) 중 하나라도 visible해야 포함
        VISIBILITY_THRESHOLD = 0.5
        visible_points = {
            i: (lm.x, lm.y)
            for i, lm in enumerate(p_res.pose_landmarks.landmark)
            if lm.visibility >= VISIBILITY_THRESHOLD
        }
        # HAND_KEYS 중 visible한 게 하나도 없으면 안 움직이는 것으로 처리
        from app.utils.analysis_utils import HAND_KEYS as _HAND_KEYS
        if any(k in visible_points for k in _HAND_KEYS):
            pose_points = visible_points
        
    return gaze_data, pose_points

# =============================================================================
# 3. Main Parallel Analysis Orchestrator
# =============================================================================

def frame_generator(video_path, stride=4):
    """비디오 프레임을 제너레이터로 반환 (메모리 절약 및 즉시 처리)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
        
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % stride != 0:
            continue
            
        # 리사이즈 없이 원본 분석 (정확도 향상)
        # 속도 저하가 우려되지만 정확한 Pose 감지가 우선
        yield frame
        
    cap.release()

def analyze_parallel(video_path, whisper_service):
    print(f"\n[Main] 병렬 분석 시작 (Optimized: Pipe + Sequential Audio): {video_path}")
    total_start_time = time.time()
    
    # 1. 오디오 추출 (선처리)
    # I/O 경합을 피하기 위해 비디오 분석 전에 수행
    print("[Main] 오디오 추출 중 (I/O 경합 방지)...")
    audio_extract_start = time.time()
    try:
        # 오디오 추출
        # Whisper 모델이 로딩되는 동안, 메인 프로세스는 놀지 않고 오디오를 추출합니다.
        audio_path = extract_audio(video_path)
    except Exception as e:
        print(f"[Main] 오디오 추출 실패: {e}")
        audio_path = None
    audio_extract_elapsed = time.time() - audio_extract_start
    print(f"[Main] 오디오 추출 완료 ({audio_extract_elapsed:.2f}s)")
    
    # 2. Whisper 분석 요청 (비동기)
    if audio_path:
        print("[Main] Whisper 서비스에 분석 요청 (Async)...")
        # Whisper에 분석 요청 (비동기)
        # 모델 로딩이 끝났다면 즉시 분석 시작, 아직 로딩 중이라면 로딩 완료 후 자동 시작
        whisper_service.transcribe_async(audio_path)
    
    # 3. 비디오 분석 (Multiprocessing Pool + Generator)
    print("[Main] 비디오 분석 시작 (스트리밍 방식)...")
    video_abs_start = time.time()
    
    speeds = []
    gazes = []
    prev_pose_points = None
    frame_cnt = 0
    
    # 영상 분석을 위한 프로세스 풀(Pool) 생성
    # processes=3: 3개의 프로세스가 동시에 프레임을 나눠서 처리
    # processes=3: FaceMesh, Pose 등을 병렬로 처리
    # 오디오 추출이 끝나는 즉시 비디오 분석 프로세스들이 가동됩니다.
    with multiprocessing.Pool(processes=3, initializer=init_worker) as pool:
        for gaze, curr_points in pool.imap(process_frame_mp, frame_generator(video_path)):
            frame_cnt += 1
            if gaze:
                gazes.append(gaze)
                
            spd = movement_speed(prev_pose_points, curr_points)
            if spd is not None:
                speeds.append(spd)
            prev_pose_points = curr_points
            
    video_abs_end = time.time()
    video_elapsed = video_abs_end - video_abs_start
    print(f"[Main] 비디오 분석 완료 ({frame_cnt} frames, {video_elapsed:.2f}s)")
    
    # 4. Whisper 결과 수신
    if audio_path:
        print("[Main] Whisper 결과 대기 중...")
        whisper_res = whisper_service.get_result()
    else:
        whisper_res = {"status": "error", "message": "Audio extraction failed"}

    if whisper_res["status"] != "success":
        print(f"Error in Whisper: {whisper_res.get('message')}")
        speech_data = {"wpm": 0, "fillers_freq": 0, "text": ""}
        audio_timing = {"init": 0, "transcribe": 0, "abs_start": 0, "abs_end": 0}
    else:
        speech_data = whisper_res["data"]
        audio_timing = whisper_res["timing"]
        
    # 5. Overlap 계산
    w_start = audio_timing.get("abs_start", 0)
    w_end = audio_timing.get("abs_end", 0)
    
    overlap_start = max(video_abs_start, w_start)
    overlap_end = min(video_abs_end, w_end)
    overlap_duration = max(0, overlap_end - overlap_start)
    
    # 전체 병렬 구간 길이 (두 작업의 합집합 기간)
    union_start = min(video_abs_start, w_start) if w_start > 0 else video_abs_start
    union_end = max(video_abs_end, w_end) if w_end > 0 else video_abs_end
    union_duration = union_end - union_start
    
    overlap_ratio = (overlap_duration / union_duration * 100) if union_duration > 0 else 0
    
    # [추가] 텍스트 스크립트 파일로 저장
    try:
        # 확장자 제거 후 _script.txt 붙이기
        script_path = os.path.splitext(video_path)[0] + "_script.txt"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(speech_data.get("text", ""))
        print(f"[Main] 스크립트 저장 완료: {script_path}")
    except Exception as e:
        print(f"[Main] 스크립트 저장 실패: {e}")
    
    # 6. 최종 결과 병합
    final_result = {
        "WPM": round(speech_data["wpm"], 1),
        "fillers_count": int(speech_data.get("fillers_count", 0)),
        "fillers_freq": round(float(speech_data.get("fillers_freq", 0.0)), 2),
        "filler_detail": speech_data.get("filler_detail", {}),
        "filler_list": speech_data.get("filler_list", []),
        "silence_count": int(speech_data.get("silence_count", 0)),
        "total_silence_sec": float(speech_data.get("total_silence_sec", 0.0)),
        "silence_ratio": float(speech_data.get("silence_ratio", 0.0)),
        "handArmMovementAvg": float(np.mean(speeds)) if speeds else 0.0,
        "handArmMovementMaxRolling": 0.0,
        "pose_warning_count": 0,
        "pose_warning_ratio": 0.0,
        "pose_rigid_count": 0,
        "pose_rigid_ratio": 0.0,
        "pose_samples": len(speeds),
        "eyes": {},
        "stt_text": speech_data.get("text", "")
    }

    if speeds:
        # --- Jitter 필터링 ---
        # static_image_mode=True 사용 시 프레임 간 트래킹이 없어
        # 실제로 안 움직여도 랜드마크 좌표가 튀는 현상(jitter) 발생.
        # 1) Dead zone: JITTER_THRESHOLD 이하 속도는 실제 움직임이 아닌 jitter로 간주 → 0 처리
        # 2) Median smoothing: 윈도우 5개 중앙값으로 스파이크 제거
        from scipy.ndimage import median_filter
        JITTER_THRESHOLD = 0.012  # 이 미만은 jitter로 간주
        speeds_arr = np.array(speeds)
        speeds_arr = np.where(speeds_arr < JITTER_THRESHOLD, 0.0, speeds_arr)
        speeds_arr = median_filter(speeds_arr, size=5)
        speeds = speeds_arr.tolist()

        MOVEMENT_WARNING_THRESHOLD = 0.025
        warning_triggers = 0
        window_size = 30

        if len(speeds) >= window_size:
            rolling_avgs = np.convolve(speeds, np.ones(window_size)/window_size, mode='valid')
            max_rolling = float(np.max(rolling_avgs))
            warning_triggers = np.sum(rolling_avgs > MOVEMENT_WARNING_THRESHOLD)
        else:
            avg_all = np.mean(speeds)
            if avg_all > MOVEMENT_WARNING_THRESHOLD:
                warning_triggers = 1
            max_rolling = float(avg_all)

        final_result["handArmMovementAvg"] = float(np.mean(speeds))
        final_result["handArmMovementMaxRolling"] = max_rolling
        final_result["pose_warning_count"] = int(warning_triggers)

        total_windows = max(1, len(speeds) - window_size + 1)
        final_result["pose_warning_ratio"] = float(round(warning_triggers / total_windows, 2))

        # 경직도: jitter 필터링 후에도 0인 프레임 비율
        rigid_frames = sum(1 for s in speeds if s < JITTER_THRESHOLD)
        final_result["pose_rigid_count"] = int(rigid_frames)
        final_result["pose_rigid_ratio"] = float(round(rigid_frames / len(speeds), 2))
    
    if gazes:
        horiz_counts = {"left":0, "center":0, "right":0}
        vert_counts  = {"up":0, "center":0, "down":0}
        for g in gazes:
            horiz_counts[g["horiz"]] += 1
            vert_counts[g["vert"]]   += 1
        
        avg_dx = float(np.mean([g["dx"] for g in gazes]))
        avg_dy = float(np.mean([g["dy"] for g in gazes]))
        
        final_result["eyes"] = {
            "avg_dx": round(avg_dx, 4),
            "avg_dy": round(avg_dy, 4),
            "horiz_mode": max(horiz_counts, key=horiz_counts.get),
            "vert_mode": max(vert_counts, key=vert_counts.get),
            "horiz_counts": horiz_counts,
            "vert_counts": vert_counts,
            "samples": len(gazes)
        }
    else:
        final_result["eyes"] = {
            "avg_dx": 0.0, "avg_dy": 0.0, 
            "horiz_mode": "n/a", "vert_mode": "n/a", 
            "samples": 0
        }
    
    total_elapsed = time.time() - total_start_time
    print(f"\n=== 분석 완료 ===")
    print(f"Total Time:       {total_elapsed:.2f}s")
    print(f"Audio Extract:    {audio_extract_elapsed:.2f}s")
    print(f"Video Analysis:   {video_elapsed:.2f}s")
    print(f"Whisper Run:      {audio_timing.get('transcribe', 0):.2f}s")
    print(f"Overlap Duration: {overlap_duration:.2f}s ({overlap_ratio:.1f}%)")
    
    print("\n=== JSON Result ===")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
    
    return final_result

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import sys, json
    from whisper import load_model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "video_standard.mp4")

    # 🔹 Whisper 모델 1회 로드
    print("🎙 Whisper 모델 로드 중...")
    service = WhisperService()
    # 1. Whisper 서비스 시작 (모델 로딩 시작 - 약 3~4초 소요)
    # 이 시점부터 Whisper 프로세스는 백그라운드에서 모델을 메모리에 올립니다.
    service.start()
    time.sleep(1) 
    print("Whisper 모델 준비 완료")

    # 🔹 분석 함수 호출 시 전달
    result = analyze_parallel(video_path, service)

    # 🔹 결과 출력 (FastAPI subprocess에서 받을 stdout)
    print(json.dumps(result, ensure_ascii=False))
    #service.stop() #테스트할때만 추가
    
