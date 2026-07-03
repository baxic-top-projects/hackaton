from __future__ import annotations

from dataclasses import replace

from .models import CalculationResult, Hypothesis, ResearchBrief


PROCESS_ALTERNATIVES = {
    "флотация": "классификация",
    "классификация": "флотация",
    "доизмельчение": "обесшламливание",
    "обесшламливание": "доизмельчение",
    "отжиг": "старение",
    "старение": "отжиг",
}


def apply_counterfactual_analysis(hypotheses: list[Hypothesis], brief: ResearchBrief) -> list[Hypothesis]:
    return [_apply_one(hypothesis, brief) for hypothesis in hypotheses]


def _apply_one(hypothesis: Hypothesis, brief: ResearchBrief) -> Hypothesis:
    baseline = _score_proxy(hypothesis, brief)
    without_main_factor = max(0.0, baseline - _factor_contribution(hypothesis))
    replacement = _replacement_scenario(hypothesis)
    replacement_score = max(0.0, baseline - 0.04 + (0.03 if replacement in brief.equipment.lower() else 0.0))
    delta_without = baseline - without_main_factor
    delta_replacement = baseline - replacement_score
    status = "ok" if delta_without >= 0.06 else "watch"
    calculation = CalculationResult(
        name="Контрфактуальный анализ",
        status=status,
        value=(
            f"baseline {baseline:.2f}; без ключевого фактора -{delta_without:.2f}; "
            f"замена на '{replacement}' {(-delta_replacement):+.2f}"
        ),
        rationale=(
            "Модуль сравнивает гипотезу с контрфактуальными сценариями: убрать основной фактор "
            "и заменить процесс ближайшей альтернативой. Это помогает объяснить, почему комбинация "
            "попала в воронку проверки."
        ),
        assumptions=[
            "оценка контрфакта использует score proxy, а не полноценный DOE",
            "альтернативы выбираются из доменной карты процессов",
        ],
    )
    rationale = (
        f"{hypothesis.rationale} CounterfactualAgent показал вклад ключевого фактора "
        f"около {delta_without:.2f} score-пункта относительно сценария без него."
    )
    confidence = min(1.0, hypothesis.confidence + 0.02 if status == "ok" else hypothesis.confidence)
    return replace(
        hypothesis,
        rationale=rationale,
        confidence=round(confidence, 3),
        calculations=[*hypothesis.calculations, calculation],
    )


def _score_proxy(hypothesis: Hypothesis, brief: ResearchBrief) -> float:
    weights = brief.weights
    return (
        hypothesis.novelty * weights.get("novelty", 0.2)
        + hypothesis.feasibility * weights.get("feasibility", 0.25)
        + hypothesis.expected_value * weights.get("expected_value", 0.3)
        + (1.0 - hypothesis.risk) * weights.get("risk", 0.15)
        + hypothesis.confidence * weights.get("confidence", 0.1)
    )


def _factor_contribution(hypothesis: Hypothesis) -> float:
    text = " ".join([hypothesis.title, hypothesis.statement, " ".join(hypothesis.tags)]).lower()
    contribution = 0.05
    if any(term in text for term in ["доизмельч", "реагент", "ph", "обесшлам"]):
        contribution += 0.06
    if any(term in text for term in ["ниобий", "молибден", "старение", "отжиг"]):
        contribution += 0.04
    return min(0.16, contribution)


def _replacement_scenario(hypothesis: Hypothesis) -> str:
    for tag in hypothesis.tags:
        if tag in PROCESS_ALTERNATIVES:
            return PROCESS_ALTERNATIVES[tag]
    return "контрольный режим"
