# TALKI - AI 기반 발표 분석 시스템

TALKI는 발표 영상을 분석하여 발표자의 **시선, 제스처, 말하기 속도 등을 분석하고 피드백을 제공하는 AI 기반 발표 분석 시스템**입니다.

영상 및 음성 데이터를 기반으로 발표자의 행동 패턴을 분석하고, 발표 습관을 정량적으로 평가하여 발표 역량을 개선할 수 있는 피드백을 제공합니다.

---

## 🧩 System Architecture

TALKI는 **Spring Boot 기반 API 서버와 FastAPI 기반 AI 분석 서버를 분리한 구조**로 설계되었습니다.

영상 업로드 및 서비스 로직은 Spring 서버에서 처리하고  
AI 분석이 필요한 데이터는 FastAPI 서버로 전달하여 분석을 수행합니다.

```
Client
   │
   ▼
Spring Boot API Server
   │
   ├── AWS S3 (영상 저장)
   │
   ├── Redis (분석 상태 관리)
   │
   ▼
FastAPI AI Analysis Server
   │
   ├── Whisper (음성 분석)
   ├── MediaPipe (시선 / 자세 분석)
   └── OpenCV (영상 처리)
   │
   ▼
Analysis Result
   │
   ▼
Spring Server → Client
```

---

## 🛠 Tech Stack

### Backend
- Spring Boot
- FastAPI

### Database / Cache
- MySQL
- Redis

### AI / Data Processing
- Whisper (Speech Recognition)
- MediaPipe (Pose / Landmark Detection)
- OpenCV (Image Processing)

### Infrastructure
- AWS EC2
- AWS S3
- Docker

### Communication
- REST API
- WebSocket

---

## 🚀 주요 기능

### 발표 영상 업로드 및 분석
사용자가 발표 영상을 업로드하면 분석 작업이 시작되며 영상은 AWS S3에 저장됩니다.

### 음성 분석
Whisper 기반 음성 인식을 통해 발표자의 **말하기 속도(WPM)** 및 발화 패턴을 분석합니다.

### 시선 분석
MediaPipe Landmark를 사용하여 발표 중 **시선 방향 및 시선 분산 여부**를 분석합니다.

### 제스처 분석
발표 중 팔과 손 움직임을 분석하여 **과도한 제스처 사용 여부**를 판단합니다.

### 실시간 피드백
WebSocket을 통해 분석 결과를 실시간으로 전달하여 발표 중 행동 패턴에 대한 피드백을 제공합니다.

---

## 📡 API Example

### 발표 분석 요청

```
POST /analyze/start
```

Request

```json
{
  "videoUrl": "https://s3.amazonaws.com/talki/video123.mp4",
  "presentationType": "online_small"
}
```

Response

```json
{
  "status": "processing",
  "presentationId": "abc123"
}
```

---

### 분석 결과 조회

```
GET /analyze/result/{presentationId}
```

Response

```json
{
  "WPM": 86.5,
  "handArmMovementAvg": 0.118,
  "pose_warning_ratio": 1.0,
  "eyes": {
    "avg_dx": 0.0026,
    "avg_dy": -0.337
  }
}
```

---

## 📂 Repository Structure

### Spring API Server
- API 서버
- 사용자 관리
- 분석 요청 처리
- WebSocket 통신

👉 [Spring Server Repository](https://github.com/JuheeNoh123/talki_spring)

### FastAPI AI Server
- 영상 / 음성 분석
- AI 모델 처리
- 분석 결과 반환

👉 [FastAPI Server Repository](https://github.com/JuheeNoh123/talki_fastAPI)

---

## 👨‍💻 My Role

- Spring Boot 기반 API 서버 설계 및 구현
- FastAPI 기반 AI 분석 서버 개발
- Redis 기반 분석 데이터 처리 구조 설계
- WebSocket 기반 실시간 분석 결과 전달 구현
- MediaPipe / Whisper 기반 분석 파이프라인 구축

---

## 💡 What I Learned

- AI 서비스에서의 **Backend + AI Server 분리 아키텍처 설계**
- 대용량 영상 데이터를 처리하기 위한 **비동기 분석 구조**
- WebSocket 기반 실시간 데이터 처리
- AI 분석 파이프라인 설계 및 서버 연동
