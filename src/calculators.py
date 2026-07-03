from __future__ import annotations

from dataclasses import dataclass, field

from .models import Hypothesis, ResearchBrief


@dataclass(frozen=True)
class CalculationResult:
    name: str
    status: str
    value: str
    rationale: str
    assumptions: list[str] = field(default_factory=list)


RELATIVE_ELEMENT_COST = {
    "никель": 1.0,
    "хром": 0.6,
    "титан": 0.9,
    "ниобий": 2.4,
    "молибден": 3.0,
    "бор": 1.7,
    "кобальт": 4.2,
    "цирконий": 2.2,
}

THERMAL_PROCESSES = {"отжиг", "старение", "закалка", "деформация"}
HYDRO_PROCESSES = {"флотация", "выщелачивание", "магнитная сепарация", "обжиг"}


def run_calculators(hypothesis: Hypothesis, brief: ResearchBrief) -> list[CalculationResult]:
    results = [
        _alloy_cost_check(hypothesis, brief),
        _equipment_feasibility_check(hypothesis, brief),
        _kpi_effect_estimate(hypothesis, brief),
    ]
    if any(tag in THERMAL_PROCESSES for tag in hypothesis.tags):
        results.append(_thermal_window_check(hypothesis, brief))
    if any(tag in HYDRO_PROCESSES for tag in hypothesis.tags):
        results.append(_process_balance_check(hypothesis, brief))
    return [result for result in results if result is not None]


def _alloy_cost_check(hypothesis: Hypothesis, brief: ResearchBrief) -> CalculationResult | None:
    element = next((tag for tag in hypothesis.tags if tag in RELATIVE_ELEMENT_COST), None)
    if element is None:
        return None
    relative_cost = RELATIVE_ELEMENT_COST[element]
    status = "ok"
    if relative_cost >= 3.5:
        status = "risk"
    elif relative_cost >= 2.0:
        status = "watch"
    if "низкий бюджет" in brief.constraints.lower() and relative_cost >= 2.0:
        status = "risk"
    return CalculationResult(
        name="Калькулятор стоимости легирования",
        status=status,
        value=f"индекс стоимости {relative_cost:.1f}x к базовому Ni",
        rationale=(
            f"Фактор '{element}' сопоставлен с относительным индексом стоимости. "
            "Это помогает отсеять гипотезы, которые улучшают KPI, но не проходят бизнес-ограничение."
        ),
        assumptions=["индексы относительные", "логистика и чистота сырья не учитываются"],
    )


def _equipment_feasibility_check(hypothesis: Hypothesis, brief: ResearchBrief) -> CalculationResult:
    equipment = brief.equipment.lower()
    matched = [tag for tag in hypothesis.tags if tag in equipment]
    status = "ok" if matched else "watch"
    value = "оборудование найдено" if matched else "нет прямого совпадения с оборудованием"
    return CalculationResult(
        name="Символьная проверка оборудования",
        status=status,
        value=value,
        rationale=(
            "Проверяет, есть ли в ограничениях лаборатории процесс или установка, "
            "необходимая для первой проверки гипотезы."
        ),
        assumptions=["используется текстовое сопоставление терминов", "производительность оборудования не оценивается"],
    )


def _kpi_effect_estimate(hypothesis: Hypothesis, brief: ResearchBrief) -> CalculationResult:
    text = " ".join([hypothesis.statement, hypothesis.mechanism, brief.target, brief.constraints]).lower()
    effect = 3.0
    if any(term in text for term in ["извлечение", "потери", "флотац", "классификац"]):
        effect += 4.0
    if any(term in text for term in ["доизмельч", "реагент", "ph", "обесшлам"]):
        effect += 2.0
    if any(term in text for term in ["жаропроч", "прочност", "старение", "отжиг"]):
        effect += 3.0
    if "низкий бюджет" in text or "без капитальной" in text:
        effect -= 1.0
    effect = max(1.0, min(12.0, effect))
    status = "ok" if effect >= 6.0 else "watch"
    return CalculationResult(
        name="Прогнозный калькулятор KPI",
        status=status,
        value=f"ожидаемый диапазон эффекта {effect - 1:.1f}-{effect + 1:.1f}% к базовому режиму",
        rationale=(
            "Оценивает первичный ожидаемый эффект по типу воздействия, целевому KPI и ограничениям. "
            "Это не заменяет эксперимент, но помогает ранжировать гипотезы перед лабораторной проверкой."
        ),
        assumptions=["модель эвристическая", "диапазон уточняется после первого DOE/скрининга"],
    )


def _thermal_window_check(hypothesis: Hypothesis, brief: ResearchBrief) -> CalculationResult:
    process = next((tag for tag in hypothesis.tags if tag in THERMAL_PROCESSES), "термообработка")
    has_furnace = any(word in brief.equipment.lower() for word in ["печь", "отжиг", "старение", "закалка"])
    status = "ok" if has_furnace else "risk"
    return CalculationResult(
        name="Физический калькулятор термоокна",
        status=status,
        value="скрининг 3 режимов: 850C, 950C, 1050C" if has_furnace else "требуется печь с контролем температуры",
        rationale=(
            f"Для процесса '{process}' предлагается минимальный DOE по температуре. "
            "Калькулятор фиксирует проверяемый диапазон вместо свободной текстовой рекомендации."
        ),
        assumptions=["температуры демонстрационные", "точный интервал уточняется по фазовой диаграмме"],
    )


def _process_balance_check(hypothesis: Hypothesis, brief: ResearchBrief) -> CalculationResult:
    process = next((tag for tag in hypothesis.tags if tag in HYDRO_PROCESSES), "процесс")
    status = "watch"
    if process in brief.equipment.lower():
        status = "ok"
    if process == "выщелачивание" and "безопас" not in brief.constraints.lower():
        status = "risk"
    return CalculationResult(
        name="Калькулятор баланса процесса",
        status=status,
        value="контроль: извлечение, выход концентрата, расход реагентов",
        rationale=(
            f"Для процесса '{process}' задается минимальный материальный баланс, "
            "чтобы гипотеза проверялась по KPI, а не только по качественному описанию."
        ),
        assumptions=["расходы реагентов задаются на лабораторном тесте", "энергозатраты оцениваются отдельно"],
    )
