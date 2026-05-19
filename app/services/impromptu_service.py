# 즉흥 말하기 분석 서비스
# 오디오 파일을 받아 추임새·발화 시간·구현 완성도·종합 피드백을 반환합니다.

from test_record_multiprocess import WhisperService
from app.services.feedback_service import clamp, calc_filler_score
from app.config.feedback_criteria import FEEDBACK_CRITERIA, PresentationType
from service_scorer import ServiceScorer

_whisper = WhisperService()
_whisper.start()

# 즉흥 말하기는 1:1 소규모 상황 → 추임새 기준이 가장 엄격한 online_small 사용
_CRITERIA = FEEDBACK_CRITERIA[PresentationType.ONLINE_SMALL]

_scorer = ServiceScorer()


def calc_completeness_score(stt_text: str, question: str, topic_desc: str, topic_tags: list) -> int:
    # 구현 완성도 점수 = 주제 적합성: 프론트엔드에서 전달받은 질문·설명·해시태그를 기반으로 ServiceScorer 점수 산출
    topic_result = _scorer.predict(
        topic_summary=question,
        topic_desc=topic_desc,
        topic_tags=topic_tags,
        doc_text=stt_text,
    )
    # ServiceScorer 반환 구조: {"scores": {"final": 0~100, ...}, ...}
    return int(topic_result.get("scores", {}).get("final", 0))


def _build_feedback(
    filler_score: int,
    completeness_score: int,
    filler_list: list,
    silence_ratio: float,
) -> list:
    # 시간 활용 / 추임새 / 주제 적합성 순서로 피드백 문장 3개를 리스트로 반환

    # 1. 시간 활용: silence_ratio 기준 (30초 녹음 기준)
    # 0.2 이하 → 침묵 6초 미만 (발화 24초+) → 충분히 활용
    # 0.4 이하 → 침묵 12초 미만 (발화 18초+) → 적절히 활용
    # 0.4 초과 → 침묵 12초 이상 → 발화 시간 부족
    if silence_ratio <= 0.2:
        time_fb = "주어진 시간을 거의 다 활용했습니다."
    elif silence_ratio <= 0.4:
        time_fb = "발화 시간을 적절히 활용했습니다."
    else:
        time_fb = "침묵 구간이 많아 발화 시간이 부족했습니다."

    # 2. 추임새: filler_list 첫 번째 항목(가장 많이 사용된 추임새)을 문장에 포함
    # filler_list 가 비어있으면 일반적인 추임새 예시("음", "아")로 대체
    top = f'"{filler_list[0]}"' if filler_list else '"음", "아"'
    if filler_score >= 70:
        filler_fb = f"{top} 같은 추임새가 적습니다."
    else:
        filler_fb = f"{top} 같은 추임새가 자주 등장했습니다."

    # 3. 주제 적합성: completeness_score 기준
    # 70점 이상 → ServiceScorer 기준 주제와의 연결이 명확한 답변
    # 50~70점   → 주제와 연결은 있지만 일부 내용이 아쉬움
    # 50점 미만  → 질문과 관련성이 낮거나 내용이 너무 부족
    if completeness_score >= 70:
        content_fb = "질문에 적절하게 답변했습니다."
    elif completeness_score >= 50:
        content_fb = "질문의 핵심 내용을 조금 더 다루면 좋겠습니다."
    else:
        content_fb = "질문과 관련된 내용을 더 구체적으로 말해보세요."

    return [time_fb, filler_fb, content_fb]


def analyze_impromptu(audio_path: str, question: str, topic_desc: str, topic_tags: list) -> dict:
    # 즉흥 말하기 오디오를 분석하여 결과를 반환합니다.
    # audio_path:  분석할 오디오 파일 경로 (WAV)
    # question:    프론트엔드에서 전달된 즉흥 질문 (주제 요약)
    # topic_desc:  질문 관련 설명
    # topic_tags:  질문 관련 해시태그 목록
    # 반환: fillers_count, fillers_freq, actual_speech_sec, completeness, feedback
    print(f"[ImpromptuService] 분석 시작 | 질문: {question[:40]}...")

    # Whisper STT 비동기 요청 후 결과 수신
    _whisper.transcribe_async(audio_path)
    whisper_res = _whisper.get_result()

    # STT 실패 시 빈 결과 반환
    if whisper_res["status"] != "success":
        error_msg = whisper_res.get("message", "알 수 없는 오류")
        print(f"[ImpromptuService] Whisper STT 실패: {error_msg}")
        return {
            "fillers_count":     0,
            "fillers_freq":      0.0,
            "actual_speech_sec": 0.0,
            "completeness":      0,
            "feedback":          [f"음성 분석 중 오류가 발생했습니다: {error_msg}"],
        }

    # whisper_res["data"] = speech_stats() 결과 (STT 텍스트 + 추임새·침묵 통계)
    stats = whisper_res["data"]

    stt_text          = stats.get("text", "")                   # Whisper 전사 텍스트
    fillers_count     = stats.get("fillers_count", 0)           # 추임새 총 횟수
    fillers_freq      = stats.get("fillers_freq", 0.0)          # 추임새 비율 (회/분)
    filler_list       = stats.get("filler_list", [])            # 감지된 추임새 단어 목록
    total_duration    = stats.get("duration", 0.0)              # 전체 오디오 길이 (초)
    total_silence_sec = stats.get("total_silence_sec", 0.0)     # 총 침묵 시간 (초)
    silence_ratio     = stats.get("silence_ratio", 0.0)         # 침묵 비율 (0.0~1.0)

    # 실제 발화 시간 = 전체 길이 - 침묵 시간
    # 2초 미만 짧은 pause는 자연스러운 발화 리듬이므로 speech_stats 에서 이미 필터링됨
    actual_speech_sec = round(max(0.0, total_duration - total_silence_sec), 1)

    # 추임새 점수: 분당 추임새 횟수가 기준치(2회/분) 이하일수록 높은 점수
    filler_score       = calc_filler_score(fillers_freq, _CRITERIA)
    # 구현 완성도: ServiceScorer로 질문·설명·해시태그 대비 답변의 주제 적합성 측정
    completeness_score = calc_completeness_score(stt_text, question, topic_desc, topic_tags)

    feedback = _build_feedback(
        filler_score       = filler_score,
        completeness_score = completeness_score,
        filler_list        = filler_list,
        silence_ratio      = silence_ratio,
    )

    print(f"[ImpromptuService] 분석 완료 | completeness: {completeness_score}, filler: {filler_score}")
    return {
        "fillers_count":     fillers_count,
        "fillers_freq":      fillers_freq,
        "actual_speech_sec": actual_speech_sec,
        "completeness":      completeness_score,
        "feedback":          feedback,
    }
