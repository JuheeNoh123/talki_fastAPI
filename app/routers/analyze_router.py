# app/routers/analyze_router.py
from fastapi import APIRouter, UploadFile, File, Query
from app.services import analyze_service_optimized as analyzer
from app.services import feedback_service
import cv2
import numpy as np
import asyncio
analysis_semaphore = asyncio.Semaphore(2)
from app.config.feedback_criteria import PresentationType
from app.schemas.analyze_schema import AnalyzeFromS3Request
import requests
import tempfile
import os
import sys
from pathlib import Path

sys.path.append("C:/Users/user/Desktop/talki_ML/Topic_model_Talki")

from service_scorer import ServiceScorer

scorer = ServiceScorer()

router = APIRouter(prefix="/analyze", tags=["Analyze"])

@router.post("/record-from-s3")
async def analyze_record(
    # presentation_type: str = Query(
    #     PresentationType.ONLINE_SMALL,
    #     description="발표 유형 (online_small | small | large)"
    # ),
    # file: UploadFile = File(...)
    req: AnalyzeFromS3Request
):
    #print(req)
    asyncio.create_task(background_analysis(req))

    return {"status": "processing"}

def download_video(url: str) -> str:
    print("📥 영상 다운로드 시작")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # 임시 파일 생성
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    # 큰 파일 대비 chunk 단위로 저장
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            tmp.write(chunk)

    tmp.close()

    print("✅ 다운로드 완료:", tmp.name)

    return tmp.name

async def background_analysis(req):

    async with analysis_semaphore:  # 🔥 동시 실행 제한
        #print(req.dict())

        print("🚀 분석 시작")
        
        video_path = download_video(req.video_url)

        loop = asyncio.get_running_loop()

        try:
            raw_result = await loop.run_in_executor(
                None,
                analyzer.analyze_record_video,
                video_path
            )
            #print(raw_result)
            text = raw_result.get("stt_text", "") #텍스트 추출
            #print("stt 텍스트",text)

            # 🔥 주제 분석
            topic_result = scorer.predict(
                topic_summary=req.topic_summary,
                topic_desc=req.topic_desc,
                topic_tags=req.topic_tags,
                doc_text=text
            )

            raw_result["topic"] = topic_result

            feedback = feedback_service.generate_feedback(
                raw_result,
                req.presentation_type
            )

            gaze = raw_result.get("eyes", {})
            raw_data = {
                "speech": {
                    "wpm": raw_result.get("WPM", 0),
                    "fillers_count": raw_result.get("fillers_count", 0),
                    "fillers_freq": raw_result.get("fillers_freq", 0.0),
                    "filler_detail": raw_result.get("filler_detail", {}),
                    "filler_list": raw_result.get("filler_list", []),
                    "silence_count": raw_result.get("silence_count", 0),
                    "total_silence_sec": raw_result.get("total_silence_sec", 0.0),
                    "silence_ratio": raw_result.get("silence_ratio", 0.0),
                    "text": raw_result.get("stt_text", ""),
                },
                "pose": {
                    "avg_speed": raw_result.get("handArmMovementAvg", 0.0),
                    "max_speed": raw_result.get("handArmMovementMaxRolling", 0.0),
                    "warning_count": raw_result.get("pose_warning_count", 0),
                    "warning_ratio": raw_result.get("pose_warning_ratio", 0.0),
                    "rigid_count": raw_result.get("pose_rigid_count", 0),
                    "rigid_ratio": raw_result.get("pose_rigid_ratio", 0.0),
                    "samples": raw_result.get("pose_samples", 0),
                },
                "gaze": {
                    "avg_dx": gaze.get("avg_dx", 0.0),
                    "avg_dy": gaze.get("avg_dy", 0.0),
                    "horizontal_mode": gaze.get("horiz_mode", ""),
                    "vertical_mode": gaze.get("vert_mode", ""),
                    "horizontal_counts": gaze.get("horiz_counts", {}),
                    "vertical_counts": gaze.get("vert_counts", {}),
                    "samples": gaze.get("samples", 0),
                    "horiz_front_ratio": round(
                        gaze.get("horiz_counts", {}).get("center", 0) / gaze.get("samples", 1)
                        if gaze.get("samples", 0) > 0 else 0.0, 3
                    ),
                    "vert_front_ratio": round(
                        gaze.get("vert_counts", {}).get("center", 0) / gaze.get("samples", 1)
                        if gaze.get("samples", 0) > 0 else 0.0, 3
                    ),
                    # front_ratio: scoring과 동일한 가중 평균 (presentation_type별 가중치 적용)
                    "front_ratio": round(
                        (lambda s=gaze.get("samples", 0),
                                 hc=gaze.get("horiz_counts", {}).get("center", 0),
                                 vc=gaze.get("vert_counts", {}).get("center", 0),
                                 pt=req.presentation_type:
                            (hc / s) * {"online_small": 0.5, "small": 0.4, "large": 0.6}.get(pt, 0.5) +
                            (vc / s) * {"online_small": 0.5, "small": 0.6, "large": 0.4}.get(pt, 0.5)
                            if s > 0 else 0.0
                        )(), 3
                    ),
                },
                "topic": raw_result.get("topic", {}),
            }

            resjson = {
                "s3_key": req.s3_key,
                "raw_data": raw_data,
                "scores": {
                    "total_score": feedback["total_score"],
                    "score_detail": feedback["score_detail"],
                },
                "llm_feedback": feedback["llm_feedback"],
            }

            requests.post(
                "http://43.201.182.246:8080/analyze/callback",
                json=resjson
            )

            print(resjson)
            # print("callback:",req.s3_key)

        finally:
            import os
            os.remove(video_path)
            print("✅ 분석 종료")
