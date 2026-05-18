# app/services/analyze_service_optimized.py
from .whisper_service import WhisperService
from test_record_multiprocess import analyze_parallel

def analyze_record_video(video_path: str):
    """녹화 영상 전체 분석 — 요청마다 독립 WhisperService 생성으로 동시 처리 안전"""
    print(f"[Analyze Service] 녹화 영상 분석 요청: {video_path}")
    service = WhisperService()
    service.start()
    try:
        return analyze_parallel(video_path, service)
    finally:
        service.stop()

