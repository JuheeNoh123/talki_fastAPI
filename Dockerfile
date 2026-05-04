# CUDA 12.1 + Python 3.11 베이스 이미지
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN python3.11 -m pip install --upgrade pip

# 작업 디렉토리
WORKDIR /app

# requirements 먼저 복사 (캐시 활용)
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# 소스코드 복사
COPY . .

# Topic_model 복사 (ML 모델 포함)
# 빌드 전에 Topic_model 폴더를 프로젝트 루트에 넣어두세요
COPY Topic_model /app/Topic_model

# 환경변수
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 포트
EXPOSE 8000

# 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]