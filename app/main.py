# app/main.py
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analyze_router, realtime_router

app = FastAPI(
    title="TALKI Realtime Feedback API",
    description="실시간 멀티모달 분석 및 피드백 제공 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드와 통신 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시엔 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(analyze_router.router, tags=["Analyze"])
app.include_router(realtime_router.router, prefix="", tags=["Realtime Analysis"])

@app.get("/")
def root():
    return {"message": "TALKI Realtime Feedback API is running"}
