from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from .models import Hypothesis, ResearchBrief


def hypotheses_to_frame(hypotheses: list[Hypothesis]) -> pd.DataFrame:
    rows = []
    for idx, hypothesis in enumerate(hypotheses, start=1):
        rows.append(
            {
                "rank": idx,
                "title": hypothesis.title,
                "total_score": hypothesis.total_score,
                "novelty": hypothesis.novelty,
                "feasibility": hypothesis.feasibility,
                "expected_value": hypothesis.expected_value,
                "risk": hypothesis.risk,
                "confidence": hypothesis.confidence,
                "sources": ", ".join(sorted({item.source for item in hypothesis.evidence})),
                "statement": hypothesis.statement,
            }
        )
    return pd.DataFrame(rows)


def hypotheses_to_json(hypotheses: list[Hypothesis], brief: ResearchBrief) -> str:
    payload = {
        "brief": asdict(brief),
        "hypotheses": [asdict(hypothesis) for hypothesis in hypotheses],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def hypotheses_to_markdown(hypotheses: list[Hypothesis], brief: ResearchBrief) -> str:
    lines = [
        "# Отчет: Фабрика гипотез",
        "",
        f"**Цель:** {brief.target}",
        f"**Ограничения:** {brief.constraints or 'не указаны'}",
        "",
        "## Ранжированные гипотезы",
        "",
    ]
    for idx, hypothesis in enumerate(hypotheses, start=1):
        lines.extend(
            [
                f"### {idx}. {hypothesis.title}",
                "",
                f"**Итоговый балл:** {hypothesis.total_score:.3f}",
                "",
                hypothesis.statement,
                "",
                f"**Механизм:** {hypothesis.mechanism}.",
                "",
                f"**Обоснование:** {hypothesis.rationale}",
                "",
                "**Риски:**",
                *[f"- {risk}" for risk in hypothesis.risks],
                "",
                "**План проверки:**",
                *[f"- {step}" for step in hypothesis.experiment_plan],
                "",
                "**Источники:**",
                *[f"- {item.source}: {item.quote}" for item in hypothesis.evidence],
                "",
            ]
        )
    return "\n".join(lines)
