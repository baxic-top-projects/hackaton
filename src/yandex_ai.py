from __future__ import annotations

import os

import requests

from .models import Hypothesis, ResearchBrief


YANDEX_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
DEFAULT_MODEL = "yandexgpt-lite/latest"


def is_yandex_configured() -> bool:
    return bool(os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"))


def generate_expert_summary(
    brief: ResearchBrief,
    hypotheses: list[Hypothesis],
    long_context: str | None = None,
) -> str:
    api_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    model = os.getenv("YANDEX_MODEL", DEFAULT_MODEL)
    timeout = int(os.getenv("YANDEX_TIMEOUT_SECONDS", "15"))
    if not api_key or not folder_id:
        return "Yandex AI Studio не настроен: задайте YANDEX_API_KEY и YANDEX_FOLDER_ID."

    payload = {
        "modelUri": f"gpt://{folder_id}/{model}",
        "completionOptions": {
            "stream": False,
            "temperature": 0.25,
            "maxTokens": "1200",
        },
        "messages": [
            {
                "role": "system",
                "text": (
                    "Ты экспертный модуль научной фабрики гипотез. "
                    "Используй GraphRAG-связи, цитаты источников и результаты калькуляторов. "
                    "Не выдумывай источники. Явно разделяй эффект, риски, расчетные ограничения "
                    "и первый лабораторный эксперимент."
                ),
            },
            {
                "role": "user",
                "text": long_context or _build_prompt(brief, hypotheses),
            },
        ],
    }
    response = requests.post(
        YANDEX_COMPLETION_URL,
        headers={"Authorization": f"Api-Key {api_key}"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    alternatives = data.get("result", {}).get("alternatives", [])
    if not alternatives:
        return "YandexGPT не вернул текстовый ответ."
    return alternatives[0].get("message", {}).get("text", "").strip()


def _build_prompt(brief: ResearchBrief, hypotheses: list[Hypothesis]) -> str:
    lines = [
        f"Цель: {brief.target}",
        f"Ограничения: {brief.constraints}",
        f"Сырье: {brief.available_materials}",
        f"Оборудование: {brief.equipment}",
        "",
        "Сформированные гипотезы:",
    ]
    for idx, hypothesis in enumerate(hypotheses[:5], start=1):
        sources = ", ".join(sorted({item.source for item in hypothesis.evidence}))
        lines.extend(
            [
                f"{idx}. {hypothesis.title}",
                f"Формулировка: {hypothesis.statement}",
                f"Механизм: {hypothesis.mechanism}",
                f"Score: {hypothesis.total_score}",
                f"Риски: {'; '.join(hypothesis.risks)}",
                f"Источники: {sources}",
            ]
        )
    lines.append(
        "Сформируй 3-5 коротких пунктов: что проверять первым, почему, какие риски контролировать."
    )
    return "\n".join(lines)
