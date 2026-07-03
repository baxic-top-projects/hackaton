from __future__ import annotations

from dataclasses import replace

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import HashingVectorizer

from .models import CalculationResult, Hypothesis, ResearchBrief


def apply_predictive_kpi_model(hypotheses: list[Hypothesis], brief: ResearchBrief) -> list[Hypothesis]:
    if not hypotheses:
        return hypotheses
    vectorizer, model = _fit_surrogate_model(hypotheses)
    adjusted = []
    for hypothesis in hypotheses:
        predicted = float(model.predict(vectorizer.transform([_hypothesis_text(hypothesis, brief)]))[0])
        expected_value = max(0.05, min(1.0, 0.65 * hypothesis.expected_value + 0.35 * predicted))
        total_score = _recalculate_total(hypothesis, brief, expected_value)
        status = "ok" if predicted >= 0.68 else "watch"
        calculation = CalculationResult(
            name="Предсказательная KPI-модель",
            status=status,
            value=f"прогноз normalized KPI uplift {predicted:.2f}; expected_value {hypothesis.expected_value:.2f}->{expected_value:.2f}",
            rationale=(
                "Легкая surrogate-модель RandomForest обучается на сгенерированных кандидатах и их "
                "оценках, затем прогнозирует относительный KPI uplift для каждой гипотезы. "
                "Это демонстрационный предиктивный слой перед лабораторным DOE."
            ),
            assumptions=[
                "модель обучается на малой выборке кандидатов текущего запуска",
                "после накопления исторических экспериментов признаки можно заменить фактическими KPI",
            ],
        )
        adjusted.append(
            replace(
                hypothesis,
                expected_value=round(expected_value, 3),
                total_score=round(total_score, 3),
                calculations=[*hypothesis.calculations, calculation],
            )
        )
    return sorted(adjusted, key=lambda item: item.total_score, reverse=True)


def _fit_surrogate_model(hypotheses: list[Hypothesis]) -> tuple[HashingVectorizer, RandomForestRegressor]:
    vectorizer = HashingVectorizer(lowercase=True, alternate_sign=False, n_features=2**12, norm="l2")
    texts = [_hypothesis_text(hypothesis, None) for hypothesis in hypotheses]
    targets = [
        max(0.05, min(1.0, hypothesis.expected_value * 0.65 + hypothesis.confidence * 0.25 + (1.0 - hypothesis.risk) * 0.1))
        for hypothesis in hypotheses
    ]
    model = RandomForestRegressor(n_estimators=48, max_depth=4, random_state=42)
    model.fit(vectorizer.transform(texts), targets)
    return vectorizer, model


def _hypothesis_text(hypothesis: Hypothesis, brief: ResearchBrief | None) -> str:
    parts = [
        hypothesis.title,
        hypothesis.statement,
        hypothesis.mechanism,
        " ".join(hypothesis.tags),
    ]
    if brief is not None:
        parts.extend([brief.target, brief.constraints, brief.equipment, brief.available_materials])
    return " ".join(parts)


def _recalculate_total(hypothesis: Hypothesis, brief: ResearchBrief, expected_value: float) -> float:
    weights = brief.weights
    return (
        hypothesis.novelty * weights.get("novelty", 0.2)
        + hypothesis.feasibility * weights.get("feasibility", 0.25)
        + expected_value * weights.get("expected_value", 0.3)
        + (1.0 - hypothesis.risk) * weights.get("risk", 0.15)
        + hypothesis.confidence * weights.get("confidence", 0.1)
    )
