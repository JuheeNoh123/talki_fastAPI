# app/routers/websocket_router.py
from fastapi import APIRouter, WebSocket
from app.services import analyze_service_landmarks as analyzer
from app.services.feedback_manager import FeedbackManager
from starlette.websockets import WebSocketDisconnect


#기본 라이브러리
import base64, cv2, numpy as np, json
import time, uuid

#레디스 (실시간 분석 결과 저장용)
from app.core.redis import redis_client

#발표 유형 설정
from app.config.feedback_criteria import PresentationType

#whisper 음성 분석 서비스
from app.services.whisper_service import whisper_service

#임시 오디오 파일 생성용
import tempfile
import wave
import os


#레디스에 구간(segment) 정보 저장하는 함수
def save_segment(presentation_id, seg_type, start, end):
    #저장할 데이터 종류
    event = { 
        "type": seg_type, # 구간 종류 (speech_fast / silence / pose_rigid / gaze_unstable)
        "start": round(start, 1), # 발표 시작 기준 시작 시간
        "end": round(end, 1), #발표 시작 기준 종료 시간
        "duration": round(end - start, 1) # 구간 길이
    }

    # Redis 리스트에 push
    redis_client.rpush(
        f"presentation:{presentation_id}:segments",
        json.dumps(event, ensure_ascii=False)
    )

    # TTL 설정 (1시간)
    redis_client.expire(
        f"presentation:{presentation_id}:segments",
        60 * 60
    )

router = APIRouter(tags=["Realtime Analysis"])

# 실시간 WebSocket 분석 엔드포인트
@router.websocket("/realtime")
async def realtime_socket(ws: WebSocket):
    # 현재 진행 중인 "구간(segment)" 상태 저장 (값이 None이면 현재 구간 없음)
    active_segments = {
        "speech_fast": None,
        "speech_slow": None, 
        "pose_rigid": None,
        "pose_unstable": None, 
        "gaze_unstable": None,
        "silence": None
    }

    
    # 정적 판단 기준
    SILENCE_THRESHOLD = 200 # 오디오 볼륨 기준
    SILENCE_LIMIT = 3.0  # 3초 이상 정적이면 구간 기록, 피드백

    # 오디오 버퍼 (STT 분석용)
    audio_buffer = []

    #프론트가 보내는 오디오 chunk 길이
    audio_chunk_duration = 1.0

    #마지막 STT 실행 시각
    last_stt_time = time.time()

    #STT 실행 간격
    STT_INTERVAL = 5.0 

    #WebSocket 연결 수락
    await ws.accept()
    print("[WebSocket] 연결 시작")

    #발표 유형 (small/large/online_small)
    presentation_type = ws.query_params.get("type", "small")
    #실시간 피드백 관리 객체
    feedback_manager = FeedbackManager(presentation_type=presentation_type) 
    #발표 ID 생성
    presentation_id = uuid.uuid4().hex
    #발표 시작 시각
    presentation_start_time = time.time()
    
    #프론트에 session_start 이벤트 전송
    await ws.send_text(json.dumps({
        "type": "session_start",
        "presentationId": presentation_id
    }, ensure_ascii=False))

    try:
        while True:
            # 현재 시각 기록 (구간 계산에 사용)
            current_time = time.time()
            data = await ws.receive_json() #프론트에서 JSON 데이터 수신
            
            #1. 랜드마크 기반 분석 (시선 + 자세)
            raw_result = analyzer.analyze_realtime_landmarks(data)

            #2. 오디오 데이터 처리
            audio_base64 = data.get("audio")
            if audio_base64:
                audio_np = analyzer.decode_audio(audio_base64) # base64-> numpy 오디오 변환
                audio_buffer.append(audio_np) # 오디오 버퍼에 저장 (STT 용)
                volume = np.abs(audio_np).mean() # 현재 오디오 볼륨 계산

                #정적 구간 감지
                if volume < SILENCE_THRESHOLD:
                    
                    #정적 시작
                    if active_segments["silence"] is None:
                        active_segments["silence"] = current_time
                    
                    if "speech" not in raw_result:
                        raw_result["speech"] = {}
                    raw_result["speech"]["silence"] = True
                else:
                    #정적 종료
                    if active_segments["silence"] is not None:
                        silence_duration = current_time - active_segments["silence"]
                        #일정 시간 이상 정적이면 segment 저장
                        if silence_duration > SILENCE_LIMIT:
                            save_segment(
                                presentation_id,
                                "silence",
                                active_segments["silence"] - presentation_start_time,
                                current_time - presentation_start_time
                            )
                    #정적 상태 종료
                    active_segments["silence"] = None

            
            
            #3. Whisper 음성 분석 실행
            speech_result = None

            if current_time - last_stt_time > STT_INTERVAL and len(audio_buffer) >= 3:
                
                # 오디오 버펴 합치기
                full_audio = np.concatenate(audio_buffer) 
                
                # 임시 wav 파일 생성
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    wav_path = tmp.name

                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(full_audio.tobytes())

                # Whisper STT 실행
                try:
                    whisper_service.transcribe_async(wav_path)
                    whisper_res = whisper_service.get_result()

                    if whisper_res["status"] == "success":
                        speech_result = whisper_res["data"]

                except Exception as e:
                    print("whisper error: ",e)
                
                finally:
                    os.remove(wav_path)
                
                # 오디오 버퍼 초기화
                audio_buffer.clear()

                # STT 실행 시각 업데이트
                last_stt_time = current_time
            
            # 4. 말 속도(빠른,느린) 구간 감지
            if speech_result:
                if "speech" not in raw_result:
                    raw_result["speech"] = {}

                raw_result["speech"].update(speech_result)

                wpm = speech_result.get("wpm", 0)

                #말 빠른 구간
                if wpm > feedback_manager.criteria["wpm_max"]:
                    #fast 시작
                    if active_segments["speech_fast"] is None:
                        active_segments["speech_fast"] = current_time - presentation_start_time

                
                    #slow 종료
                    if active_segments["speech_slow"] is not None:

                        save_segment(
                            presentation_id,
                            "speech_slow",
                            active_segments["speech_slow"],
                            current_time - presentation_start_time
                        )

                        active_segments["speech_slow"] = None

                    

                #말 느린 구간
                elif wpm < feedback_manager.criteria["wpm_min"] and wpm > 0:
                    #slow 시작
                    if active_segments["speech_slow"] is None:
                        active_segments["speech_slow"] = current_time - presentation_start_time
                
                
                    #fast 종료
                    if active_segments["speech_fast"] is not None:

                        save_segment(
                            presentation_id,
                            "speech_fast",
                            active_segments["speech_fast"],
                            current_time - presentation_start_time
                        )

                        active_segments["speech_fast"] = None
                
                #정상 속도 구간
                else:
                    # fast 종료
                    if active_segments["speech_fast"] is not None:
                        save_segment(
                            presentation_id,
                            "speech_fast",
                            active_segments["speech_fast"],
                            current_time - presentation_start_time
                        )
                        active_segments["speech_fast"] = None

                    # slow 종료
                    if active_segments["speech_slow"] is not None:
                        save_segment(
                            presentation_id,
                            "speech_slow",
                            active_segments["speech_slow"],
                            current_time - presentation_start_time
                        )
                        active_segments["speech_slow"] = None
                    

            # 5. 시선 자세 피드백 업데이트            
            manager_feedback = feedback_manager.update(raw_result)

            # 6. 자세 경직,산만 구간 감지
            pose_landmarks = raw_result.get("pose_landmarks")

            if pose_landmarks:

                avg_speed = np.mean(feedback_manager.movement_speeds) if feedback_manager.movement_speeds else 0
                #자세 경직 구간
                if avg_speed < feedback_manager.criteria["pose_min"]:

                    if active_segments["pose_rigid"] is None:
                        active_segments["pose_rigid"] = current_time

                else:

                    if active_segments["pose_rigid"] is not None:

                        save_segment(
                            presentation_id,
                            "pose_rigid",
                            active_segments["pose_rigid"] - presentation_start_time,
                            current_time - presentation_start_time
                        )

                        active_segments["pose_rigid"] = None
                
                #자세 산만 구간
                if avg_speed > feedback_manager.criteria["pose_max"]:

                    if active_segments["pose_unstable"] is None:
                        active_segments["pose_unstable"] = current_time

                else:

                    if active_segments["pose_unstable"] is not None:

                        save_segment(
                            presentation_id,
                            "pose_unstable",
                            active_segments["pose_unstable"] - presentation_start_time,
                            current_time - presentation_start_time
                        )

                        active_segments["pose_unstable"] = None
           
           # 7. 시선 불안정 구간 감지
            if raw_result.get("gaze_unstable"):

                if active_segments["gaze_unstable"] is None:
                    active_segments["gaze_unstable"] = current_time

            else:

                if active_segments["gaze_unstable"] is not None:

                    save_segment(
                        presentation_id,
                        "gaze_unstable",
                        active_segments["gaze_unstable"] - presentation_start_time,
                        current_time - presentation_start_time
                    )

                    active_segments["gaze_unstable"] = None


            # 8. 프론트에 실시간 피드백 전송
            await ws.send_text(json.dumps({
                "type": "feedback",
                "raw_result": raw_result,
                "data": manager_feedback
            }, ensure_ascii=False))


    except WebSocketDisconnect:
        print("클라이언트 나감")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WebSocket] 연결 종료: {e}")

    # 9. 발표 종료 시 열려있는 segment 정리
    finally:
        now = time.time()

        for seg_type, start_time in active_segments.items():

            if start_time is not None:

                save_segment(
                    presentation_id,
                    seg_type,
                    start_time - presentation_start_time,
                    now - presentation_start_time
                )   
        
        if ws.client_state.name != 'DISCONNECTED':
            await ws.close()
        
