# 원문 기반 요약 말하기 분석 서비스
# 원문을 제시하고 사용자가 요약하여 말하면, 핵심 키워드 포함 여부 및 연결 자연스러움을 분석합니다.

from test_record_multiprocess import WhisperService
from service_scorer import ServiceScorer

_whisper = WhisperService()
_whisper.start()

_scorer = ServiceScorer()


def _detect_keywords(stt_text: str, keywords: list) -> list:
    # STT 텍스트에 각 키워드가 포함되어 있는지 확인
    return [kw for kw in keywords if kw in stt_text]


def calc_naturalness_score(stt_text: str, original_summary: str, original_text: str, original_tags: list) -> int:
    # 연결 자연스러움 점수 = ServiceScorer로 원문 대비 요약 발화의 논리적 구성·적합도 측정
    topic_result = _scorer.predict(
        topic_summary=original_summary,
        topic_desc=original_text,
        topic_tags=original_tags,
        doc_text=stt_text,
    )
    # ServiceScorer 반환 구조: {"scores": {"final": 0~100, ...}, ...}
    return int(topic_result.get("scores", {}).get("final", 0))


def _naturalness_label(score: int) -> str:
    # 80점 이상 → 우수, 60~79 → 양호, 40~59 → 보통, 40 미만 → 미흡
    if score >= 80:
        return "우수"
    elif score >= 60:
        return "양호"
    elif score >= 40:
        return "보통"
    return "미흡"


def _build_feedback(
    keyword_count: int,
    naturalness_score: int,
    keywords_missing: list,
) -> list:
    # 1. 키워드 포함도
    if keyword_count == 3:
        keyword_fb = "원문의 핵심 내용을 잘 포함하여 요약했습니다."
    elif keyword_count == 2:
        keyword_fb = "원문의 일부 핵심 내용이 포함되었습니다."
    else:
        keyword_fb = "원문의 핵심 내용이 많이 빠져 있습니다."

    # 2. 연결 자연스러움: naturalness_score 기준 
    # 70점 이상 → 요약 내용의 논리적 흐름과 문장 품질 양호
    # 50~70점   → 연결이 다소 어색
    # 50점 미만  → 논리적 흐름 개선 필요
    if naturalness_score >= 70:
        natural_fb = "내용 간 논리적 연결이 자연스럽습니다."
    elif naturalness_score >= 50:
        natural_fb = "요약 내용의 연결이 다소 어색한 부분이 있었습니다."
    else:
        natural_fb = "핵심 내용을 논리적으로 연결하는 연습이 필요합니다."

    # 3. 미포함 키워드 안내
    if not keywords_missing:
        overall_fb = "원문의 핵심 키워드를 모두 담아 완성도 높은 요약이었습니다."
    elif len(keywords_missing) == 1:
        overall_fb = f'"{keywords_missing[0]}" 키워드를 포함하면 더 완성도 높은 요약이 됩니다.'
    else:
        overall_fb = f'"{keywords_missing[0]}", "{keywords_missing[1]}" 키워드를 포함하면 더 완성도 높은 요약이 됩니다.'

    return [keyword_fb, natural_fb, overall_fb]


def analyze_summary(
    audio_path: str,
    keywords: list,
    original_summary: str,
    original_text: str,
    original_tags: list,
) -> dict:
    # 원문 기반 요약 말하기 오디오를 분석하여 결과를 반환합니다.
    # audio_path:       분석할 오디오 파일 경로 (WAV)
    # keywords:         원문 핵심 키워드 목록
    # original_summary: 원문 요약
    # original_text:    원문 전체
    # original_tags:    원문 관련 해시태그 목록
    # 반환: keywords_used, keyword_count, naturalness_label, feedback
    print(f"[SummaryService] 분석 시작 | 원문 길이: {len(original_text)}자, 키워드: {keywords}")

    # Whisper STT 비동기 요청 후 결과 수신
    _whisper.transcribe_async(audio_path)
    whisper_res = _whisper.get_result()

    # STT 실패 시 빈 결과 반환
    if whisper_res["status"] != "success":
        error_msg = whisper_res.get("message", "알 수 없는 오류")
        print(f"[SummaryService] Whisper STT 실패: {error_msg}")
        return {
            "keywords_used":     [],
            "keyword_count":     0,
            "naturalness_label": "미흡",
            "feedback":          [f"음성 분석 중 오류가 발생했습니다: {error_msg}"],
        }

    # whisper_res["data"] = speech_stats() 결과 (STT 텍스트 + 추임새·침묵 통계)
    stats    = whisper_res["data"]
    stt_text = stats.get("text", "")  # Whisper 전사 텍스트

    # 핵심 키워드 기준으로 요약 발화에 포함된 키워드 감지
    keywords_used    = _detect_keywords(stt_text, keywords)
    keywords_missing = [kw for kw in keywords if kw not in keywords_used]
    keyword_count    = len(keywords_used)

    # 연결 자연스러움: 원문 요약·원문·해시태그를 기준으로 ServiceScorer 논리적 구성·적합도 측정
    naturalness_score = calc_naturalness_score(stt_text, original_summary, original_text, original_tags)
    # 우수/양호/보통/미흡
    naturalness_label = _naturalness_label(naturalness_score)

    feedback = _build_feedback(
        keyword_count     = keyword_count,
        naturalness_score = naturalness_score,
        keywords_missing  = keywords_missing,
    )

    print(f"[SummaryService] 분석 완료 | 키워드 {keyword_count}/{len(keywords)}, naturalness: {naturalness_label}({naturalness_score})")
    return {
        "keywords_used":     keywords_used,
        "keyword_count":     keyword_count,
        "naturalness_label": naturalness_label,
        "feedback":          feedback,
    }
