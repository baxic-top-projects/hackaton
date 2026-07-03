from __future__ import annotations

import itertools
import re
from collections import Counter

from .models import Evidence, Hypothesis, ResearchBrief
from .retrieval import KnowledgeIndex


ADDITIVES = {
    "ниобий": "формирования дисперсных карбидов и стабилизации зеренной структуры",
    "молибден": "твердорастворного упрочнения и повышения устойчивости к ползучести",
    "титан": "измельчения зерна и связывания азота/углерода в стабильные фазы",
    "бор": "упрочнения границ зерен при малых концентрациях",
    "цирконий": "очистки границ зерен и снижения склонности к межкристаллитному разрушению",
    "хром": "повышения коррозионной и окалиностойкости",
    "никель": "стабилизации аустенитной матрицы и повышения пластичности",
    "кобальт": "повышения жаропрочности матрицы",
}

PROCESSES = {
    "отжиг": "контролируемого выделения вторичных фаз",
    "закалка": "фиксации пересыщенного твердого раствора",
    "старение": "образования наноразмерных упрочняющих частиц",
    "деформация": "накопления дислокаций и последующей рекристаллизации",
    "флотация": "селективного извлечения ценных минералов",
    "обжиг": "изменения фазового состава и раскрытия минералов",
    "выщелачивание": "перевода целевого компонента в раствор",
    "магнитная сепарация": "разделения фаз по магнитной восприимчивости",
}

PROPERTIES = {
    "жаропрочность": "прочность при повышенной температуре",
    "прочность": "предел прочности и сопротивление разрушению",
    "твердость": "сопротивление пластической деформации",
    "пластичность": "деформационная способность без разрушения",
    "извлечение": "доля целевого металла в концентрате",
    "себестоимость": "стоимость сырья и операций",
    "коррозионная стойкость": "устойчивость к агрессивной среде",
}

RISK_HINTS = {
    "ниобий": "рост стоимости шихты и риск образования крупных карбидов",
    "молибден": "удорожание легирования и возможная сегрегация",
    "бор": "узкое технологическое окно дозирования",
    "флотация": "чувствительность к pH, реагентному режиму и тонкости помола",
    "выщелачивание": "требования к безопасности реагентов и очистке растворов",
    "старение": "риск переcтаривания и потери пластичности",
}


def generate_hypotheses(
    brief: ResearchBrief,
    index: KnowledgeIndex,
    limit: int = 8,
) -> list[Hypothesis]:
    query = " ".join(
        [
            brief.target,
            brief.constraints,
            brief.available_materials,
            brief.equipment,
        ]
    )
    seed_evidence = index.search(query, limit=10)
    corpus = " ".join(e.quote.lower() for e in seed_evidence) + " " + query.lower()

    additives = _rank_terms(corpus, ADDITIVES, fallback=["ниобий", "молибден", "титан"])
    processes = _rank_terms(corpus, PROCESSES, fallback=["отжиг", "старение", "флотация"])
    properties = _rank_terms(corpus, PROPERTIES, fallback=[_target_property(brief.target)])

    candidates: list[Hypothesis] = []
    for additive, process, prop in itertools.islice(
        itertools.product(additives, processes, properties),
        max(limit * 3, 12),
    ):
        hypothesis_query = f"{brief.target} {additive} {process} {prop} {brief.constraints}"
        evidence = index.search(hypothesis_query, limit=4) or seed_evidence[:3]
        scores = _score(additive, process, prop, evidence, brief)
        mechanism = _mechanism(additive, process)
        candidates.append(
            Hypothesis(
                title=f"{additive.capitalize()} + {process}: влияние на {prop}",
                statement=(
                    f"Если применить {process} с контролем режима и использовать {additive} "
                    f"как ключевой фактор воздействия, то можно улучшить {prop} за счет {mechanism}."
                ),
                mechanism=mechanism,
                rationale=_rationale(additive, process, prop, evidence, brief),
                novelty=scores["novelty"],
                feasibility=scores["feasibility"],
                expected_value=scores["expected_value"],
                risk=scores["risk"],
                confidence=scores["confidence"],
                total_score=scores["total_score"],
                evidence=evidence,
                experiment_plan=_experiment_plan(additive, process, prop, brief),
                risks=_risks(additive, process, brief),
                resources=_resources(process, brief),
                tags=[additive, process, prop],
            )
        )

    unique = _deduplicate(candidates)
    return sorted(unique, key=lambda item: item.total_score, reverse=True)[:limit]


def _rank_terms(corpus: str, dictionary: dict[str, str], fallback: list[str]) -> list[str]:
    counts = Counter()
    for term in dictionary:
        counts[term] = len(re.findall(rf"\b{re.escape(term)}\b", corpus, flags=re.IGNORECASE))
    ranked = [term for term, count in counts.most_common() if count > 0]
    for term in fallback:
        if term and term in dictionary and term not in ranked:
            ranked.append(term)
    return ranked[:4]


def _target_property(target: str) -> str:
    target_lower = target.lower()
    for prop in PROPERTIES:
        if prop in target_lower:
            return prop
    return "прочность"


def _mechanism(additive: str, process: str) -> str:
    additive_part = ADDITIVES.get(additive, "изменения фазового состава")
    process_part = PROCESSES.get(process, "контроля технологического режима")
    return f"{additive_part} при одновременном эффекте {process_part}"


def _score(
    additive: str,
    process: str,
    prop: str,
    evidence: list[Evidence],
    brief: ResearchBrief,
) -> dict[str, float]:
    evidence_strength = min(1.0, sum(item.score for item in evidence) / max(len(evidence), 1) * 2.2)
    constraints = brief.constraints.lower()
    novelty = 0.72
    novelty += 0.08 if additive not in constraints else -0.04
    novelty += 0.05 if len({item.source for item in evidence}) > 1 else 0

    feasibility = 0.68
    feasibility += 0.12 if process in brief.equipment.lower() else 0
    feasibility += 0.08 if additive in brief.available_materials.lower() else 0
    feasibility -= 0.1 if "низкий бюджет" in constraints or "минимальный бюджет" in constraints else 0

    expected_value = 0.7
    expected_value += 0.12 if prop in brief.target.lower() else 0
    expected_value += 0.08 if any(word in brief.target.lower() for word in ["15%", "20%", "снизить", "повысить"]) else 0

    risk = 0.34
    risk += 0.1 if additive in {"молибден", "кобальт", "ниобий"} else 0
    risk += 0.08 if process in {"выщелачивание", "обжиг"} else 0
    risk -= 0.06 if process in brief.equipment.lower() else 0

    confidence = 0.45 + evidence_strength * 0.45

    weights = brief.weights
    total = (
        novelty * weights.get("novelty", 0.2)
        + feasibility * weights.get("feasibility", 0.25)
        + expected_value * weights.get("expected_value", 0.3)
        + (1 - risk) * weights.get("risk", 0.15)
        + confidence * weights.get("confidence", 0.1)
    )
    return {
        "novelty": _bounded(novelty),
        "feasibility": _bounded(feasibility),
        "expected_value": _bounded(expected_value),
        "risk": _bounded(risk),
        "confidence": _bounded(confidence),
        "total_score": _bounded(total),
    }


def _rationale(
    additive: str,
    process: str,
    prop: str,
    evidence: list[Evidence],
    brief: ResearchBrief,
) -> str:
    sources = ", ".join(sorted({item.source for item in evidence})) or "загруженной базе знаний"
    return (
        f"Цель '{brief.target}' связана с показателем '{prop}'. В источниках ({sources}) "
        f"найдены фрагменты о факторах '{additive}' и '{process}'. Поэтому гипотеза "
        "помещается в начало воронки как проверяемая комбинация состава, режима и KPI."
    )


def _experiment_plan(additive: str, process: str, prop: str, brief: ResearchBrief) -> list[str]:
    return [
        f"Зафиксировать базовый состав/режим и измерить исходный KPI: {prop}.",
        f"Подготовить 3-4 варианта с разным уровнем фактора '{additive}' в допустимых ограничениях.",
        f"Провести '{process}' на доступном оборудовании: {brief.equipment or 'лабораторная установка'}.",
        "Сравнить результаты с контролем, оценить доверительный интервал и воспроизводимость.",
        "Принять решение: масштабировать, уточнить диапазон факторов или закрыть гипотезу.",
    ]


def _risks(additive: str, process: str, brief: ResearchBrief) -> list[str]:
    risks = [
        RISK_HINTS.get(additive, "неопределенность механизма влияния на промышленном сырье"),
        RISK_HINTS.get(process, "чувствительность результата к режиму и подготовке образцов"),
    ]
    if brief.budget:
        risks.append(f"Бюджетное ограничение: {brief.budget}.")
    if brief.constraints:
        risks.append(f"Доменные ограничения: {brief.constraints}.")
    return risks


def _resources(process: str, brief: ResearchBrief) -> list[str]:
    resources = ["лабораторные образцы", "протокол измерения целевого KPI", "журнал экспериментов"]
    if brief.available_materials:
        resources.append(f"доступное сырье: {brief.available_materials}")
    if brief.equipment:
        resources.append(f"оборудование: {brief.equipment}")
    elif process in {"отжиг", "старение", "закалка"}:
        resources.append("печь термообработки и средства контроля температуры")
    return resources


def _deduplicate(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    seen: set[tuple[str, ...]] = set()
    unique: list[Hypothesis] = []
    for hypothesis in hypotheses:
        key = tuple(hypothesis.tags)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hypothesis)
    return unique


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)
