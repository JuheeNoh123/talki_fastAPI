# -*- coding: utf-8 -*-
# app/services/gaze_service.py
# 시선 분석 서비스
# - analyze_gaze_video(video_path, dot_positions): 발표 영상 전체 시선 분석

import cv2
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from app.utils.analysis_utils import gaze_from_landmarks

# 화면을 가로 3구역으로만 나눔: 좌(0~33%) / 중(33~67%) / 우(67~100%)
# 세로는 pitch 추정이 부정확해서 제외
#----------------------------------------------
# 세로 제외 이유 추가 주석
# 고개 끄덕임이나 카메라 각도, 얼굴 기울기에 따라 흔들림이 커서 오차가 많음
#----------------------------------------------
_NX_ZONES = 3

# 머리 방향 → 화면 % 좌표 변환 스케일
# 값이 클수록 작은 움직임도 화면에 크게 반영
# 현재 모두 정답인 영상 => 100점 나오도록 조정된 상태
# 적당히 정면과 아래, 오른쪽 살짝 본 영상은 53.3점 나옴
_HEAD_YAW_SCALE   = 1000.0
_HEAD_PITCH_SCALE = 375.0
_IRIS_SCALE       = 200.0

def _head_gaze_to_screen(lms, iris_dx: float, iris_dy: float) -> tuple:
    """코끝-눈 중점 오프셋(머리 방향) + iris 편차 → 화면 % 좌표 (gx, gy)"""
    # 랜드마크 좌표 추출.
    # MediaPipe FaceMesh 랜드마크에서 코끝(1번)과 왼눈 끝(33번), 오른눈 끝(263번) 좌표를 꺼냄
    # 두 눈 끝점의 중간값으로 눈 중점을 계산
    nose_x    = lms[1].x
    nose_y    = lms[1].y
    eye_mid_x = (lms[33].x + lms[263].x) / 2
    eye_mid_y = (lms[33].y + lms[263].y) / 2
    # 얼굴 크기 정규화
    # 얼굴너비와 높이 나눠서 정규화
    # 1e-6은 0으로 나누는걸 방지하는 아주 작은 값
    face_w    = abs(lms[263].x - lms[33].x) + 1e-6
    face_h    = abs(lms[152].y - lms[10].y)  + 1e-6

    # 고개 방향 계산
    head_yaw   = (nose_x - eye_mid_x) / face_w
    head_pitch = (nose_y - eye_mid_y) / face_h

    # 화면 % 좌표로 변환.
    # 정면(50%)을 기준점으로 고개 방향 + 홍채 편차를 더해서 화면 좌표로 변환
    gx = 50.0 + head_yaw * _HEAD_YAW_SCALE + iris_dx * _IRIS_SCALE
    gy = 50.0 + head_pitch * _HEAD_PITCH_SCALE + iris_dy * _IRIS_SCALE

    # 계산 결과가 0~100% 범위를 벗어나지 않도록 강제로 잘라냄
    return max(0.0, min(100.0, gx)), max(0.0, min(100.0, gy))


def _to_zone(x_percent: float) -> int:
    """화면 x% → 수평 구역 인덱스 (0=좌, 1=중, 2=우)"""
    return min(int(x_percent / 100 * _NX_ZONES), _NX_ZONES - 1)


def analyze_gaze_video(video_path: str, dot_positions: list) -> dict:
    """
    발표 영상 전체를 분석해서 시선 점수 반환.
    가로 3구역(좌/중/우)으로 매핑.

    Args:
        video_path   : 발표 영상 경로
        dot_positions: [{"x": float, "y": float}, ...] 화면 % 좌표 (청중 위치)

    Returns:
        {
            "gaze_score"      : float,  # 0~100점
            "coverage"        : float,  # 커버된 구역 비율 (0.0~1.0)
            "covered_zones"   : int,    # 방문한 구역 수
            "total_zones"     : int,    # dot이 있는 전체 구역 수
            "forward_ratio"   : float,  # 화면을 본 프레임 비율 (0.0~1.0)
            "missed_zones"    : list,   # 방문하지 않은 구역 이름 목록 (예: ["좌", "우"])
            "most_missed_zone": str,    # 놓친 구역 중 dot이 가장 많은 쪽 (없으면 None)
        }
        실패 시 None 반환
    """
    _ZONE_NAMES = {0: "좌", 1: "중", 2: "우"}

    # 프론트에서 받은 dot 좌표들을 좌/중/우 구역으로 분류.
    # 구역별 개수 셈.
    # dict으로 각 구역마다 dot이 몇개 위치하고 있는지 저장해둠.
    zone_dot_counts: dict[int, int] = {}
    for d in dot_positions:
        z = _to_zone(d["x"])
        zone_dot_counts[z] = zone_dot_counts.get(z, 0) + 1

    # dot_zones는 dot이 존재하는 구역 종류 집합
    # dot이 골고루 좌/중/우에 있다는 보장은 없기 때문에, dot이 없는 구역을 안봤다고 감점시키는 상황을 방지하기 위함.
    dot_zones = set(zone_dot_counts.keys())
    total_zones = len(dot_zones)
    visited_zones = set()

    # MediaPipe FaceMesh 설정
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        # 홍채 랜드마크를 활성화 -> gaze_from_landmarks가 눈동자 좌표 사용 가능
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 영상 파일 프레임 단위로 읽을 수 있게 준비
    cap = cv2.VideoCapture(video_path)
    # 파일 경로 잘못 or 손상된 영상 => 진행 X 조기 종료.
    if not cap.isOpened():
        print(f"[GazeService] 영상 열기 실패: {video_path}")
        # 이때 이미 초기화한 face_mesh도 같이 닫아줘야 메모리 누수 발생 안 함.
        # 때문에 먼저 닫고 None 반환
        face_mesh.close()
        return None

    # 영상 프레임 루프
    # 30fps 영상이면 초당 6프레임 간격/60fps 영상이면 초당 12프레임 간격으로
    # 건너뛰어서 초당 5번 분석
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps / 5))  

    total_frames = 0
    forward_frames = 0
    frame_idx = 0

    print(f"[GazeService] 분석 시작 | FPS: {fps:.1f} | dot 구역 수: {total_zones}")

    while True:
        # 영상에서 프레임 하나 읽음. (초당 5번 분석한다 했던 프레임 중 하나)
        # 더 읽을 프레임 없으면 루프 종료.
        ret, frame = cap.read()
        if not ret:
            break
        
        # 위 설정해둔 거 여기서 실행 => 6프레임마다 1번만 분석(30fps 기준). 나머지는 건너뜀.
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        total_frames += 1
        # OpenCV는 BGR 포맷이라 mediapipe용 RGB로 변환 후 얼굴 랜드마크 추출
        # 얼굴 안잡히면 건너뜀.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            continue
        
        # 시선 -> 구역 판정
        # 얼굴이 잡힌 프레임은 카메라(청중)를 보고 있는 것으로 카운트.
        # 홍채+고개 방향으로 x좌표 계산 -> 구역 변환 -> dot있는 구역이면 방문 기록
        forward_frames += 1

        lms = result.multi_face_landmarks[0].landmark
        # 홍채 + 고개 방향으로 x좌표 계산
        gaze = gaze_from_landmarks(lms)
        gx, _ = _head_gaze_to_screen(lms, gaze["dx"], gaze["dy"])

        # 구역 변환
        zone = _to_zone(gx)
        # dot이 있는 구역이면 방문 기록
        if zone in dot_zones:
            visited_zones.add(zone)

    # 확인 다 해서 영상 파일이랑 faceMesh 닫기
    cap.release()
    face_mesh.close()

    # 검사할 수 있는 프레임이 없을때 출력
    if total_frames == 0:
        print("[GazeService] 분석 가능한 프레임 없음")
        return None
    
    # 방문한 구역 수/전체 dot 구역 수, 좌/중/우 3구역 다봤으면 1.0, 하나도 안봤음 0.0
    covered_count = len(visited_zones)
    coverage = covered_count / total_zones if total_zones > 0 else 0.0
    # 분석한 프레임 중 얼굴이 잡힌 비율. 카메라를 얼마나 봤는지
    forward_ratio = forward_frames / total_frames

    # 구역 커버리지 70점 + 화면 응시 30점 
    gaze_score = round(coverage * 70 + forward_ratio * 30, 1)

    # 놓친 구역 계산
    missed_zones = dot_zones - visited_zones
    missed_zone_names = [_ZONE_NAMES[z] for z in sorted(missed_zones)]
    # 놓친 구역 중 dot이 가장 많은 쪽
    most_missed_zone = _ZONE_NAMES[max(missed_zones, key=lambda z: zone_dot_counts[z])] if missed_zones else None

    print(f"[GazeService] 완료 | 방문 구역: {sorted(visited_zones)} | coverage: {coverage:.1%} ({covered_count}/{total_zones}) | forward: {forward_ratio:.1%} | 점수: {gaze_score}")
    print(f"[GazeService] 놓친 구역: {missed_zone_names} | 가장 많이 놓친 쪽: {most_missed_zone}")

    return {
        "gaze_score"      : gaze_score,
        "coverage"        : round(coverage, 4),
        "covered_zones"   : covered_count,
        "total_zones"     : total_zones,
        "forward_ratio"   : round(forward_ratio, 4),
        "missed_zones"    : missed_zone_names,
        "most_missed_zone": most_missed_zone,
    }
