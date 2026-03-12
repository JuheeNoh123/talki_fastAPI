from collections import deque
import time
import numpy as np
from app.utils.analysis_utils import movement_speed
from app.config.feedback_criteria import (
    PresentationType,
    FEEDBACK_CRITERIA,
)

class FeedbackManager:
    def __init__(self, presentation_type: PresentationType.SMALL):
        self.presentation_type = presentation_type
        self.criteria = FEEDBACK_CRITERIA[presentation_type]

        # --- 시선(Gaze) 상태 ---
        self.gaze_horiz_buffer = deque(maxlen=30)
        self.gaze_vert_buffer  = deque(maxlen=30)
        self.last_gaze_feedback_time = 0

        # --- 자세(Pose) 상태 ---
        self.prev_pose_landmarks = None
        self.movement_speeds = deque(maxlen=30) # 약 3초
        self.last_pose_feedback_time = 0

        # --- 음성(Speech) 상태 (누적) ---
        self.speech_start_time = None 
        self.wpm_buffer = deque(maxlen=5)      # WPM 이동 평균용 (최근 5번 업데이트)
        self.filler_buffer = deque(maxlen=5)   # 추임새 빈도 이동 평균용
        self.last_speech_feedback_time = 0
        
        # --- 쿨다운 (재알림 방지) ---
        self.COOLDOWN_SEC = 5.0 

    def update(self, result: dict) -> str | None:
        """
        새로운 프레임 결과로 상태를 업데이트하고, 피드백 메시지가 발생하면 반환합니다.
        피드백이 필요 없는 경우 None을 반환합니다.
        """
        current_time = time.time()
        feedback_messages = []
    
        # 1. 시선 업데이트
        gaze = result.get("gaze")
        if gaze:
            horiz = gaze.get("horiz")
            vert  = gaze.get("vert")

            self.gaze_horiz_buffer.append(horiz if horiz else "N/A")
            self.gaze_vert_buffer.append(vert if vert else "N/A")
        else:
            self.gaze_horiz_buffer.append("N/A")
            self.gaze_vert_buffer.append("N/A")
    
        # =====================
        # 2. 시선 판단 (수평 + 수직 동일)
        # =====================
        if (
            current_time - self.last_gaze_feedback_time > self.COOLDOWN_SEC
            and len(self.gaze_horiz_buffer) >= 10   # 테스트용 완화
        ):
            front_count = self.gaze_horiz_buffer.count("center")
            front_ratio = front_count / len(self.gaze_horiz_buffer)

            if front_ratio < self.criteria["gaze_front_ratio"]:
                feedback_messages.append(
                    "정면을 바라보는 시간이 부족합니다. 청중을 더 자주 바라봐주세요."
                )
                self.last_gaze_feedback_time = current_time
                result["gaze_unstable"] = True
            else:
                result["gaze_unstable"] = False
            

        # 2. 자세 업데이트 (움직임 속도)
        curr_landmarks = result.get("pose_landmarks")
        if curr_landmarks:
            if self.prev_pose_landmarks:
                # 공통 로직으로 이동 거리(속도) 계산
                speed = movement_speed(self.prev_pose_landmarks, curr_landmarks)
                if speed is not None:
                    self.movement_speeds.append(speed)
            self.prev_pose_landmarks = curr_landmarks
        
        # 자세 체크 (평균 속도가 너무 빠르거나 느린 경우 트리거)
        if current_time - self.last_pose_feedback_time > self.COOLDOWN_SEC and len(self.movement_speeds) >= 15:
            avg_speed = np.mean(self.movement_speeds)
            # print(f"[DEBUG] Avg Speed: {avg_speed:.4f}") # 디버깅용
            
            if avg_speed > self.criteria["pose_max"]:
                 feedback_messages.append("몸을 너무 많이 움직이고 있습니다. 조금 더 차분한 자세를 취해보세요.")
                 self.last_pose_feedback_time = current_time
            elif avg_speed < self.criteria["pose_min"]:
                 feedback_messages.append("자세가 다소 경직되어 있습니다. 자연스러운 제스처를 사용해 보세요.")
                 self.last_pose_feedback_time = current_time
        
        # 3. 음성 업데이트
        speech = result.get("speech")

        if speech and speech.get("silence"):
            feedback_messages.append("발표 중 정적이 길어졌습니다. 이어서 말씀해 보세요.")
        if speech and speech.get("text"):
            wpm = speech.get("wpm", 0)
            fillers = speech.get("fillers_freq", 0)
            
            self.wpm_buffer.append(wpm)
            self.filler_buffer.append(fillers)

            # 버퍼가 어느 정도 찼을 때만 피드백 (e.g. 최소 1개 이상이면 동작하되, 평균값 사용)
            if current_time - self.last_speech_feedback_time > self.COOLDOWN_SEC and len(self.wpm_buffer) > 0:
                avg_wpm = sum(self.wpm_buffer) / len(self.wpm_buffer)
                avg_fillers = sum(self.filler_buffer) / len(self.filler_buffer)
                
                # WPM 피드백
                if avg_wpm > self.criteria["wpm_max"]:
                    feedback_messages.append("말이 다소 빠릅니다. 조금 천천히 말씀해 보세요.")
                    self.last_speech_feedback_time = current_time
                    self.wpm_buffer.clear() # 피드백 후 버퍼 초기화 (새로운 흐름 측정)
                elif avg_wpm < self.criteria["wpm_min"] and avg_wpm > 0:
                    feedback_messages.append("말이 다소 느립니다. 자신감 있게 말씀해 보세요.")
                    self.last_speech_feedback_time = current_time
                    self.wpm_buffer.clear()
                
                # 추임새 피드백
                if avg_fillers > self.criteria["fillers_per_min"]:
                    feedback_messages.append("습관적인 추임새(음, 어)가 들립니다. 의식적으로 줄여보세요.")
                    self.last_speech_feedback_time = current_time # WPM과 쿨다운 공유 (너무 많은 메시지 방지)
                    self.filler_buffer.clear()
                
        print(feedback_messages)
        if feedback_messages:
            return " / ".join(feedback_messages)
        
        return None
