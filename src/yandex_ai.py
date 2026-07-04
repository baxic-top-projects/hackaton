from __future__ import annotations

import os

import requests

from .models import Hypothesis, ResearchBrief


YANDEX_COMPLETION_URL = "https://ai.api.cloud.yandex.net/v1/responses"
DEFAULT_MODEL = "aliceai-llm"


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
        return "Alice AI LLM не настроена: задайте YANDEX_API_KEY и YANDEX_FOLDER_ID."

    payload = {
        "model": f"gpt://{folder_id}/{model}",
        "temperature": 0.25,
        "max_output_tokens": 1200,
        "input": _build_responses_input(long_context or _build_prompt(brief, hypotheses)),
    }
    response = requests.post(
        YANDEX_COMPLETION_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Project": folder_id,
            "X-Folder-Id": folder_id,
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code in {401, 403}:
        detail = response.text[:800] if response.text else "empty response"
        raise RuntimeError(
            "Alice AI LLM запретила запрос. Проверьте, что API key активен, относится к нужному облаку, "
            "имеет доступ к Alice AI LLM / Yandex AI Studio, а YANDEX_FOLDER_ID совпадает с каталогом ключа. "
            f"HTTP {response.status_code}: {detail}"
        )
    response.raise_for_status()
    data = response.json()
    text = _extract_responses_text(data)
    if not text:
        return "Alice AI LLM не вернула текстовый ответ."
    return text


def _build_responses_input(user_text: str) -> list[dict[str, str]]:
    system_text = (
        "Ты экспертный модуль научной фабрики гипотез. "
        "Используй GraphRAG-связи, цитаты источников и результаты калькуляторов. "
        "Не выдумывай источники. Явно разделяй эффект, риски, расчетные ограничения "
        "и первый лабораторный эксперимент."
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def _extract_responses_text(data: dict) -> str:
    if data.get("output_text"):
        return str(data["output_text"]).strip()
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


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
