def build_feedback_prompt(tags: dict) -> str:
    m = tags["metrics"]
    stt_text = tags.get("stt_text", "").strip()

    # --- 기준 충족 여부 (점수 기반 판단: 80점 이상이면 장점으로 인정) ---
    # 이진 임계값 체크 대신 점수로 판단해야 경계값 근처 억울한 케이스 방지
    scores = tags.get("score_detail", {})
    gaze_ok   = scores.get("gaze", 0) >= 80
    pose_ok   = scores.get("pose", 0) >= 80
    wpm_ok    = scores.get("speech_speed", 0) >= 80
    filler_ok = scores.get("fillers", 0) >= 80
    topic_ok  = scores.get("topic", 0) >= 70

    # --- 시선 자연어 해석 (LLM에게 수치 대신 해석된 문장 전달) ---
    h = m["gaze_horiz_front_ratio"]
    v = m["gaze_vert_front_ratio"]

    horiz_mode = m.get("gaze_horiz_mode", "")
    horiz_dir = "왼쪽" if horiz_mode == "left" else ("오른쪽" if horiz_mode == "right" else "정면")

    if h >= 0.8:
        horiz_desc = "좌우 시선은 정면을 잘 유지했습니다."
    elif h >= 0.5:
        horiz_desc = f"시선이 {horiz_dir}으로 가끔 벗어났습니다."
    else:
        horiz_desc = f"발표 내내 시선이 {horiz_dir}으로 크게 치우쳤습니다."

    vert_mode = m.get("gaze_vert_mode", "")
    vert_dir = "위쪽" if vert_mode == "up" else ("아래쪽" if vert_mode == "down" else "정면")

    if v >= 0.6:
        vert_desc = "상하 시선도 정면을 잘 유지했습니다."
    elif v >= 0.3:
        vert_desc = f"시선이 {vert_dir}으로 자주 벗어났습니다."
    else:
        vert_desc = f"발표 내내 거의 {vert_dir}을 바라봤습니다."

    # --- 제스처 자연어 해석 ---
    # pose_strength: 장점 섹션용 (긍정 표현만)
    # pose_detail: 자세/제스처 분석 섹션용 (세부 진단 포함)
    spd = m["pose_avg_speed"]
    rigid = m["pose_rigid_ratio"]
    if spd > m["pose_max"]:
        pose_strength = None
        pose_detail = "팔이나 몸이 과도하게 많이 움직여 산만한 인상을 줄 수 있습니다."

    elif spd < m["pose_min"]:
        pose_strength = None
        if rigid > 0.7:
            pose_detail = "발표 내내 팔과 몸을 거의 움직이지 않아 경직된 인상을 줬습니다."
        else:
            pose_detail = "전반적으로 움직임이 적어 표현이 제한적이었습니다."

    else:
        if rigid > 0.5:
        # 경직 심함
            pose_strength = "쓸데없는 제스처 없이 자세가 안정적으로 유지되어 발표 내용에 집중할 수 있는 환경을 만들었습니다."
            pose_detail = "자세가 산만하지 않아 발표에 집중할 수 있었지만, 움직임이 거의 없어 전달력이 아쉬웠습니다. 핵심 내용을 강조할 때 자연스러운 제스처를 추가하면 더 좋은 전달력을 갖출 수 있습니다."

        elif rigid > 0.2:
            # ✅ 수정 포인트: rigid=0.31이 여기 해당
            pose_strength = "쓸데없는 제스처 없이 자세가 차분하게 유지되어 청중이 발표 내용에 집중할 수 있었습니다."
            pose_detail = "자세가 산만하지 않아 발표에 집중할 수 있지만, 자연스러운 제스처를 적절히 추가하면 전달력이 한층 높아질 것입니다."

        else:
            # 정상 움직임 (경직 거의 없음)
            pose_strength = "제스처가 과하지 않으면서도 자연스럽게 활용되어 안정적인 인상을 줬습니다."
            pose_detail = "제스처와 움직임이 적절한 수준으로 고르게 유지됐습니다."

    # 프롬프트에는 pose_detail 전달 (분석용), 장점 판단은 pose_ok + pose_strength로
    pose_desc = pose_detail

    # --- 말 속도 자연어 해석 ---
    if wpm_ok:
        wpm_desc = "말 속도가 적절했습니다."
    elif m["speech_wpm"] > m["wpm_max"]:
        wpm_desc = "말이 너무 빠르게 전달되어 청중이 따라가기 어려울 수 있습니다."
    else:
        wpm_desc = "말이 너무 느려 청중의 집중력이 떨어질 수 있습니다."

    # --- 추임새 자연어 해석 ---
    if filler_ok:
        filler_desc = "기준 이하 (언급 금지)"
    else:
        flist = ", ".join(m["filler_list"]) if m["filler_list"] else "확인된 추임새 있음"
        filler_desc = f"허용 기준을 초과했으며, '{flist}' 등의 추임새가 반복 사용됐습니다."

    # --- 주제 자연어 해석 ---
    if topic_ok:
        topic_desc = "주제와의 연결이 명확했습니다."
    elif m["topic_score"] >= 50:
        topic_desc = "주제와 연결은 있지만 문장 표현이 명확하지 않습니다."
    else:
        topic_desc = "주제와의 연결이 전반적으로 약해 핵심 메시지가 잘 전달되지 않았습니다."

    return f"""
너는 발표 코칭 전문가다.
아래에 이미 해석된 발표 진단 내용이 주어진다.
이 해석을 바탕으로, 사용자가 읽기 쉬운 자연스러운 한국어로 피드백을 작성하라.

[핵심 규칙]
- 수치, 비율, 프레임 수를 절대 그대로 언급하지 말 것
- "0.067", "119프레임 중" 같은 표현 금지
- 아래 해석된 문장을 기반으로 자연스럽게 풀어 쓸 것
- ex) "좌우 시선은 안정적이었으나, 발표 내내 거의 위쪽을 바라봐 청중과의 눈 맞춤이 이루어지지 않았습니다."

[진단 결과 요약]
- 시선 (좌우): {horiz_desc}
- 시선 (상하): {vert_desc}
- 제스처: {pose_desc}
- 말 속도: {wpm_desc}
- 추임새: {filler_desc}
- 주제 적합성: {topic_desc} (관련성: {m["topic_relevance"]}점 / 문장품질: {m["topic_quality"]}점)
  - 논리 흐름 문제 문장 수: {m["low_coherence_count"]}
  - 주제 이탈 문장 수: {m["off_topic_count"]}
  - 가장 주제와 벗어난 문장: "{tags.get("topic_focus", "없음")}"
- 침묵 구간: {m["silence_count"]}회 ({m["total_silence_sec"]}초)

[발화 내용 앞부분]
{stt_text if stt_text else "없음"}

[집중 개선 대상]
{tags["key_focus"]}

[기준 충족 현황 - 장점 판단 기준]
- 시선: {"✅" if gaze_ok else "❌"}
- 제스처: {"✅ → 장점 표현: " + pose_strength if (pose_ok and pose_strength) else "❌ (장점 언급 금지)"}
- 말 속도: {"✅" if wpm_ok else "❌"}
- 추임새: {"✅" if filler_ok else "❌"}
- 주제: {"✅" if topic_ok else "❌"}

[절대 규칙]
1. ✅ 항목만 장점으로 언급. ❌ 항목은 절대 장점으로 쓰지 말 것.
   ✅ 가 하나도 없으면 "장점" 항목에 "이번 발표에서 기준을 충족한 항목이 없습니다." 명시.
   제스처 ✅ 이면 반드시 "장점 표현" 문장을 그대로 사용할 것. "제한적", "부족" 같은 부정어 절대 금지.
2. 추임새가 "기준 이하 (언급 금지)"이면 어느 항목에서도 추임새를 언급하지 말 것.
3. 침묵=0 이고 추임새 기준 초과이면 → "추임새로 발화를 채운 것"으로 해석. "자연스러운 흐름" 금지.
4. 위로성 표현 금지: "그래도", "기본기는 있어요", "앞으로 나아질 거예요" 등 금지.
5. 데이터가 나쁘면 그대로 진단. 왜곡하거나 부드럽게 포장하지 말 것.
6. "주제 적합성 분석 결과" 항목에서 "가장 주제와 벗어난 문장"이 있으면 반드시 해당 문장을 직접 인용(따옴표 포함)하여 구체적으로 언급할 것.
   예: '어, 오늘은, 음...' 과 같은 문장은 주제와 직접적인 연관이 없어 발표 흐름을 끊었습니다.
7. 각 value는 5문장 이상 10문장 이하. JSON 외 텍스트 출력 금지.
8. 제스처 ✅ 이고 rigid가 감지된 경우, 장점 표현은 반드시 "산만하지 않음/쓸데없는 제스처 없음" 각도로 작성하고, 자세/제스처 분석에서는 "제스처 추가 시 전달력 향상" 방향으로 작성할 것.
"""
