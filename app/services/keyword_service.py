# 키워드 말하기 분석 서비스
# 주어진 키워드의 사용 여부와 발화 연결 자연스러움을 분석합니다.

from test_record_multiprocess import WhisperService
from service_scorer import ServiceScorer

_whisper = WhisperService()
_whisper.start()

_scorer = ServiceScorer()


def _detect_keywords(stt_text: str, keywords: list) -> list:
    # STT 텍스트에 각 키워드가 포함되어 있는지 확인
    return [kw for kw in keywords if kw in stt_text]


def calc_naturalness_score(stt_text: str, keywords: list) -> int:
    # 연결 자연스러움 점수 = ServiceScorer로 논리적 구성·문장 적합도 측정
    # 키워드를 주제로 전달하여 발화의 논리적 연결성과 문장 품질 분석
    joined = ", ".join(keywords)
    topic_result = _scorer.predict(
        topic_summary=f"{joined}에 관한 발화",
        topic_desc=f"다음 키워드를 모두 포함하여 논리적으로 연결된 문장을 말해야 합니다: {joined}",
        topic_tags=keywords,
        doc_text=stt_text,
    )
    # ServiceScorer 반환 구조: {"scores": {"final": 0~100, ...}, ...}
    return int(topic_result.get("scores", {}).get("final", 0))


def _naturalness_label(score: int) -> str:
    # naturalness_score를 프론트엔드 표시용 라벨로 변환
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

    # 1. 키워드 활용도: 키워드는 항상 3개 고정이므로 개수로 직접 분기
    # 3개 → 전부 사용, 2개 → 대부분 사용, 1개 이하 → 자연스럽게 녹이는 연습 권유
    if keyword_count == 3:
        keyword_fb = "대부분의 키워드를 자연스럽게 사용했습니다."
    elif keyword_count == 2:
        keyword_fb = "일부 키워드를 사용했습니다."
    else:
        keyword_fb = "키워드를 문장에 자연스럽게 녹이는 연습이 필요합니다."

    # 2. 연결 자연스러움: naturalness_score 기준 (ServiceScorer 점수)
    # 70점 이상 → 논리적 구성과 문장 품질 모두 양호
    # 50~70점   → 연결이 다소 어색
    # 50점 미만  → 논리적 흐름 개선 필요
    if naturalness_score >= 70:
        natural_fb = "키워드 간 논리적 연결이 좋습니다."
    elif naturalness_score >= 50:
        natural_fb = "키워드 연결이 다소 어색한 부분이 있었습니다."
    else:
        natural_fb = "키워드 간 논리적 흐름을 더 다듬어 보세요."

    # 3. 미사용 키워드 안내: 빠진 키워드를 이름으로 직접 언급
    if not keywords_missing:
        overall_fb = "모든 키워드를 활용하여 완성도가 높습니다."
    elif len(keywords_missing) == 1:
        overall_fb = f'"{keywords_missing[0]}" 키워드를 추가하면 더 완성도가 높아집니다.'
    else:
        overall_fb = f'"{keywords_missing[0]}", "{keywords_missing[1]}" 키워드를 추가하면 더 완성도가 높아집니다.'

    return [keyword_fb, natural_fb, overall_fb]


def analyze_keyword(audio_path: str, keywords: list) -> dict:
    # 키워드 말하기 오디오를 분석하여 결과를 반환합니다.
    # audio_path: 분석할 오디오 파일 경로 (WAV)
    # keywords:   프론트엔드에서 전달된 키워드 목록
    # 반환: keyword_count, naturalness_label, keywords_used, feedback
    print(f"[KeywordService] 분석 시작 | 키워드: {keywords}")

    # Whisper STT 비동기 요청 후 결과 수신
    _whisper.transcribe_async(audio_path)
    whisper_res = _whisper.get_result()

    # STT 실패 시 빈 결과 반환
    if whisper_res["status"] != "success":
        error_msg = whisper_res.get("message", "알 수 없는 오류")
        print(f"[KeywordService] Whisper STT 실패: {error_msg}")
        return {
            "keyword_count":     0,
            "naturalness_label": "미흡",
            "keywords_used":     [],
            "feedback":          [f"음성 분석 중 오류가 발생했습니다: {error_msg}"],
        }

    # whisper_res["data"] = speech_stats() 결과 (STT 텍스트 + 추임새·침묵 통계)
    stats = whisper_res["data"]

    stt_text = stats.get("text", "")  # Whisper 전사 텍스트

    # STT 텍스트에서 실제로 사용된 키워드 목록과 미사용 키워드 목록 추출
    keywords_used    = _detect_keywords(stt_text, keywords)
    keywords_missing = [kw for kw in keywords if kw not in keywords_used]
    keyword_count    = len(keywords_used)

    # 연결 자연스러움: ServiceScorer로 논리적 구성·문장 적합도 측정
    naturalness_score = calc_naturalness_score(stt_text, keywords)
    # 우수/양호/보통/미흡
    naturalness_label = _naturalness_label(naturalness_score)

    feedback = _build_feedback(
        keyword_count     = keyword_count,
        naturalness_score = naturalness_score,
        keywords_missing  = keywords_missing,
    )

    print(f"[KeywordService] 분석 완료 | 키워드 {keyword_count}/3, naturalness: {naturalness_label}({naturalness_score})")
    return {
        "keyword_count":     keyword_count,
        "naturalness_label": naturalness_label,
        "keywords_used":     keywords_used,
        "feedback":          feedback,
    }
