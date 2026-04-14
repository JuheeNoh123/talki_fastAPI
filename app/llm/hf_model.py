# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import pipeline
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)





def translate_to_korean(text: str) -> str:
    if not text:
        return ""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            # 🔒 규칙: JSON 강제
            {
                "role": "system",
                "content": (
                    "너는 발표 코칭 전문가다.\n"
                    "반드시 아래 JSON 형식으로만 출력하라.\n"
                    "JSON 이외의 텍스트는 절대 출력하지 말 것.\n"
                    "모든 key는 반드시 포함해야 한다.\n"
                    "각 value는 5문장 이상 10문장 이하로 작성하라.\n"
                    "{\n"
                    '  "장점": "",\n'
                    '  "성장 포인트": "",\n'
                    '  "연습": "",\n'
                    '  "음성 분석 결과": "",\n'
                    '  "반복어 분석 결과": "",\n'
                    '  "시선 분석 결과": "",\n'
                    '  "자세/제스처 분석 결과": "",\n'
                    '  "주제 적합성 분석 결과": "",\n'
                    '  "전체 분석 결과": ""\n'
                    "}\n\n"
                    "[작성 조건]\n"
                    "- 모든 값은 한국어 문장으로 작성\n"
                    "- 비판하지 말고 개선 중심\n"
                    "- 실제 사람이 코칭하듯 따뜻한 말투\n"
                    "- 각 value는 5문장 이상\n"
                    "- 숫자는 직접 언급하지 말 것\n"
                    "- 한 가지 개선 포인트만 강조"
                )
            },

            # ✅ JSON 예시 (매우 중요)
            {
                "role": "assistant",
                "content": (
                    "{\n"
                    '  "장점": "전반적으로 차분한 발화와 안정적인 시선 처리로 신뢰감 있는 인상을 주었어요.",\n'
                    '  "성장 포인트": "시선이 한쪽에 머무는 경향이 있어 청중과의 연결을 더 넓히면 전달력이 좋아질 수 있어요.",\n'
                    '  "연습": "문장이 바뀔 때마다 시선을 자연스럽게 이동하는 연습을 해보세요.",\n'
                    '  "음성 분석 결과": "말의 속도와 리듬은 전반적으로 안정적으로 유지되었어요.",\n'
                    '  "반복어 분석 결과": "반복어 사용은 소량으로 발표 흐름을 방해하지는 않았어요.",\n'
                    '  "시선 분석 결과": "카메라를 비교적 잘 바라보며 안정적인 인상을 주고 있었어요.",\n'
                    '  "자세/제스처 분석 결과": "몸의 움직임이 조심스러워 핵심 포인트에서 제스처를 더하면 좋아 보여요.",\n'
                    '  "주제 적합성 분석 결과": "발표 주제와의 연결이 전반적으로 유지되고 있었으나 일부 문장에서 흐름이 끊기는 부분이 있었어요.",\n'
                    '  "전체 분석 결과": "기본기가 잘 갖춰진 발표였고 시선과 제스처를 보완하면 더 완성도 높아질 수 있어요."\n'
                    "}"
                )
            },

            # 🧠 실제 입력
            {
                "role": "user",
                "content": text
            }
        ]
    )

    return response.output_text.strip()
