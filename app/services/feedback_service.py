# app/services/feedback_service.py
import numpy as np
import json
from app.config.feedback_criteria import (
    PresentationType,
    FEEDBACK_CRITERIA,
)
from app.llm.prompt_builder import build_feedback_prompt
#from app.llm.hf_model import generate_feedback2
from app.llm.hf_model import translate_to_korean

def clamp(score: float) -> int:
    return int(max(0, min(100, score)))


def calc_gaze_score(horiz_ratio: float, vert_ratio: float, criteria: dict) -> int:
    std = criteria["gaze_front_ratio"]

    def axis_score(r: float) -> float:
        if r >= std:
            return 100.0
        return max(0.0, 100.0 - ((std - r) / std) * 100.0)

    h_score = axis_score(horiz_ratio)
    v_score = axis_score(vert_ratio)

    # 수직(상하)에 가중치를 더 줌: 위를 많이 보면 점수가 확실히 떨어져야 함
    # online_small/small: 카메라/청중 눈높이 응시가 핵심 → 수직 60%
    return clamp(int(h_score * 0.4 + v_score * 0.6))


def calc_wpm_score(wpm: float, criteria: dict) -> int:
    min_wpm = criteria["wpm_min"]
    max_wpm = criteria["wpm_max"]

    if wpm == 0:
        return 0
    if min_wpm <= wpm <= max_wpm:
        return 100

    if wpm < min_wpm:
        diff = (min_wpm - wpm) / min_wpm
    else:
        diff = (wpm - max_wpm) / max_wpm

    return clamp(100 - diff * 100)


def calc_filler_score(fillers: float, criteria: dict) -> int:
    allowed = criteria["fillers_per_min"]
    if fillers <= allowed:
        return 100
    # 초과 비율 기반 감점: 허용치 대비 몇 배 초과했는지
    # 허용치의 2배 → 약 50점, 3배 → 약 25점 수준으로 완만하게 감점
    excess_ratio = (fillers - allowed) / max(allowed, 1)
    return clamp(int(100 / (1 + excess_ratio)))


def calc_pose_score(avg_speed: float, rigid_ratio: float, criteria: dict, presentation_type: str) -> int:
    min_s = criteria["pose_min"]
    max_s = criteria["pose_max"]

    if min_s <= avg_speed <= max_s:
        return 100

    if avg_speed > max_s:
        # 과잉 움직임: 모든 유형 공통으로 큰 패널티
        # diff*100 대신 softer curve 적용 (기준 2배 초과 시 약 40점대)
        diff = (avg_speed - max_s) / max_s
        return clamp(100 - diff * 35)
    else:
        # 경직: 발표 유형별 차등 패널티
        # online_small은 카메라 앞이라 경직이 자연스러움 → 거의 패널티 없음
        if presentation_type == "online_small":
            return clamp(100 - rigid_ratio * 5)
        elif presentation_type == "small":
            return clamp(100 - rigid_ratio * 15)
        else:  # large: 넓은 공간에서 경직은 더 부자연스러움
            return clamp(100 - rigid_ratio * 25)

def interpret_topic(topic_result: dict) -> dict:
    scores = topic_result.get("scores", {})
    final = scores.get("final", 0)

    keywords = topic_result.get("keywords", [])
    evidence = topic_result.get("evidence", {})
    sentence_analysis = topic_result.get("sentence_analysis", [])
    worst = topic_result.get("worst_sentence", {})

    low_coherence_count = sum(
        1 for s in sentence_analysis if "low_coherence" in s.get("flags", [])
    )
    off_topic_count = len(evidence.get("off_topic_sentences", []))

    # on_topic 플래그 제외, final 점수 기반으로만 판단
    if final < 40:
        topic_label = "주제와의 연결이 매우 약한 발표"
    elif final < 60:
        topic_label = "주제와의 연결이 약한 발표"
    elif final < 75:
        topic_label = "주제는 유지되지만 명확성이 부족한 발표"
    else:
        topic_label = "주제 전달이 명확한 발표"

    return {
        "label": topic_label,
        "keywords": keywords[:3],
        "off_topic_count": off_topic_count,
        "low_coherence_count": low_coherence_count,
        "worst_sentence": worst.get("sentence", ""),
    }

def find_worst_metric(metrics):
    diffs = {
        "gaze": abs(metrics["gaze_diff"]),
        "pose": abs(metrics["pose_diff"]),
        "speech": abs(metrics["wpm_diff"]),
        "filler": abs(metrics["fillers_diff"]),
    }
    return max(diffs, key=diffs.get)
        
def derive_tags(score_detail, metrics, criteria, total_score, topic_result):
    tags = {}

    topic_info = interpret_topic(topic_result)

    tags["topic"] = topic_info["label"]
    tags["topic_keywords"] = topic_info["keywords"]
    tags["topic_problem"] = (
        "논리 흐름이 일부 끊깁니다"
        if topic_info["low_coherence_count"] > 2
        else "안정적입니다"
    )
    tags["topic_focus"] = topic_info["worst_sentence"]

    tags["total_score"] = total_score

    tags["key_focus"] = find_worst_metric(metrics)

    return tags

def generate_feedback(analysis_result: dict, presentation_type: str):
    criteria = FEEDBACK_CRITERIA[presentation_type]
    feedback = []

    # --- 시선 피드백 ---
    gaze = analysis_result.get("eyes", {})
    horiz_counts = gaze.get("horiz_counts", {})
    vert_counts = gaze.get("vert_counts", {})
    samples = gaze.get("samples", 0)

    horiz_front_ratio = horiz_counts.get("center", 0) / samples if samples > 0 else 0.0
    vert_front_ratio = vert_counts.get("center", 0) / samples if samples > 0 else 0.0

    # 발표 유형별 수평/수직 가중치
    # online_small: 카메라 정면이 핵심 → 수평/수직 동등
    # small: 청중이 눈높이, 수직이 약간 더 중요
    # large: 청중이 넓게 퍼져 수직 편차 허용 → 수직 가중치 낮춤
    GAZE_WEIGHTS = {
        "online_small": (0.5, 0.5),
        "small":        (0.4, 0.6),
        "large":        (0.6, 0.4),
    }
    w_horiz, w_vert = GAZE_WEIGHTS.get(presentation_type, (0.5, 0.5))
    front_ratio = horiz_front_ratio * w_horiz + vert_front_ratio * w_vert

    # # --- 자세/동작 피드백 ---
    avg_speed = analysis_result.get("handArmMovementAvg", 0.0)
    rigid_ratio = analysis_result.get("pose_rigid_ratio", 0.0)

    # # --- 음성 피드백 ---
    wpm = analysis_result.get("WPM", 0)
    fillers = analysis_result.get("fillers_freq", 0)

    # --- 종합 요약 ---
    gaze_score = calc_gaze_score(horiz_front_ratio, vert_front_ratio, criteria)
    speech_score = calc_wpm_score(wpm, criteria)
    filler_score = calc_filler_score(fillers, criteria)
    pose_score = calc_pose_score(avg_speed, rigid_ratio, criteria, presentation_type)
    topic_score = analysis_result.get("topic", {}).get("scores", {}).get("final", 0)
    
    total_score = round(
        ((gaze_score + pose_score) / 2) * 0.5 +
        ((speech_score + filler_score) / 2) * 0.3 +
        topic_score * 0.2
    )

    topic_result = analysis_result.get("topic", {})
    topic_info = interpret_topic(topic_result)

    topic_relevance = topic_result.get("scores", {}).get("topic", 0)
    topic_quality = topic_result.get("scores", {}).get("quality", 0)

    score_detail = {
        "gaze": gaze_score,
        "speech_speed": speech_score,
        "fillers": filler_score,
        "pose": pose_score,
        "topic": topic_score,
        "topic_relevance": topic_relevance,
        "topic_quality": topic_quality,
    }

    # LLM 프롬프트 생성용 내부 metrics/tags (반환값에는 포함하지 않음)
    internal_metrics = {
        "gaze_front_ratio": round(front_ratio, 2),
        "gaze_horiz_front_ratio": round(horiz_front_ratio, 2),
        "gaze_vert_front_ratio": round(vert_front_ratio, 2),
        "gaze_vert_mode": gaze.get("vert_mode", ""),
        "gaze_horiz_mode": gaze.get("horiz_mode", ""),
        "gaze_threshold": criteria["gaze_front_ratio"],
        "gaze_diff": round(front_ratio - criteria["gaze_front_ratio"], 3),

        "pose_avg_speed": round(avg_speed, 4),
        "pose_min": criteria["pose_min"],
        "pose_max": criteria["pose_max"],
        "pose_diff": round(
            avg_speed - criteria["pose_max"] if avg_speed > criteria["pose_max"]
            else criteria["pose_min"] - avg_speed if avg_speed < criteria["pose_min"]
            else 0,
            4
        ),

        "speech_wpm": round(wpm, 1),
        "wpm_min": criteria["wpm_min"],
        "wpm_max": criteria["wpm_max"],
        "wpm_diff": round(
            wpm - criteria["wpm_max"] if wpm > criteria["wpm_max"]
            else criteria["wpm_min"] - wpm if wpm < criteria["wpm_min"]
            else 0,
            1
        ),

        "speech_fillers": fillers,
        "fillers_limit": criteria["fillers_per_min"],
        "fillers_diff": fillers - criteria["fillers_per_min"],
        "filler_list": analysis_result.get("filler_list", []),

        "topic_score": topic_score,
        "topic_relevance": topic_relevance,
        "topic_quality": topic_quality,
        "off_topic_count": topic_info["off_topic_count"],
        "low_coherence_count": topic_info["low_coherence_count"],

        "silence_count": analysis_result.get("silence_count", 0),
        "total_silence_sec": analysis_result.get("total_silence_sec", 0.0),
        "silence_ratio": analysis_result.get("silence_ratio", 0.0),

        "pose_rigid_count": analysis_result.get("pose_rigid_count", 0),
        "pose_rigid_ratio": analysis_result.get("pose_rigid_ratio", 0.0),
    }
    internal_tags = derive_tags(score_detail, internal_metrics, criteria, total_score, topic_result)
    internal_tags["metrics"] = internal_metrics
    internal_tags["score_detail"] = score_detail
    internal_tags["stt_text"] = analysis_result.get("stt_text", "")[:500]

    prompt = build_feedback_prompt(internal_tags)
    print(prompt)
    #english_feedback = generate_feedback2(prompt)
    #print(english_feedback)
    raw = translate_to_korean(prompt)
    print(raw)
    try:
        llm_feedback = json.loads(raw)
    except json.JSONDecodeError:
        llm_feedback = {
            "장점": "",
            "성장 포인트": "",
            "연습": "",
            "음성 분석 결과": "",
            "반복어 분석 결과": "",
            "시선 분석 결과": "",
            "자세/제스처 분석 결과": "",
            "주제 적합성 분석 결과": "",
            "전체 분석 결과": ""
        }

    return {
        "total_score": total_score,
        "score_detail": score_detail,
        "llm_feedback": llm_feedback,
    }



