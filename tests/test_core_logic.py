import unittest

from src.calculators import run_calculators
from src.metadata import extract_metadata
from src.models import Hypothesis, ResearchBrief
from src.text_normalization import append_normalized_terms


def make_hypothesis(*, tags: list[str], statement: str = "", mechanism: str = "") -> Hypothesis:
    return Hypothesis(
        title="Candidate process",
        statement=statement,
        mechanism=mechanism,
        rationale="Evidence-based rationale",
        novelty=0.8,
        feasibility=0.7,
        expected_value=0.7,
        risk=0.2,
        confidence=0.7,
        total_score=0.7,
        evidence=[],
        experiment_plan=[],
        risks=[],
        resources=[],
        tags=tags,
    )


class TextNormalizationTests(unittest.TestCase):
    def test_translates_distinct_english_and_chinese_terms_once(self) -> None:
        normalized = append_normalized_terms("Flotation tailings and 尾矿 require regrinding.")

        self.assertIn("флотация", normalized)
        self.assertIn("доизмельчение", normalized)
        self.assertEqual(normalized.count("хвосты"), 1)

    def test_does_not_match_english_terms_inside_larger_words(self) -> None:
        text = "The recoveryman label should remain untouched."
        self.assertEqual(append_normalized_terms(text), text)


class MetadataTests(unittest.TestCase):
    def test_extracts_dates_author_conditions_and_language_flags(self) -> None:
        metadata = extract_metadata(
            "Автор: Иван Петров. Отчет 2025-03-14: pH 7.5, 950°C, выход 82%. 浮选",
            "reports/trial.pdf",
            {"batch": "A-17"},
        )

        self.assertEqual(metadata["extension"], ".pdf")
        self.assertEqual(metadata["batch"], "A-17")
        self.assertIn("2025-03-14", metadata["dates"])
        self.assertEqual(metadata["authors"], ["Иван Петров"])
        self.assertIn("pH 7.5", metadata["conditions"]["ph"])
        self.assertIn("950°C", metadata["conditions"]["temperature"])
        self.assertTrue(metadata["has_chinese"])
        self.assertTrue(metadata["has_latin"])


class CalculatorTests(unittest.TestCase):
    def test_low_budget_flags_expensive_alloy_and_missing_furnace(self) -> None:
        hypothesis = make_hypothesis(
            tags=["кобальт", "старение"],
            statement="Повысить жаропрочность сплава",
        )
        brief = ResearchBrief(
            target="Повысить прочность",
            constraints="Низкий бюджет",
            available_materials="Сплав",
            equipment="Микроскоп",
            budget="Низкий",
            weights={},
        )

        results = {result.name: result for result in run_calculators(hypothesis, brief)}

        self.assertEqual(results["Калькулятор стоимости легирования"].status, "risk")
        self.assertEqual(results["Физический калькулятор термоокна"].status, "risk")
        self.assertEqual(results["Символьная проверка оборудования"].status, "watch")


if __name__ == "__main__":
    unittest.main()
