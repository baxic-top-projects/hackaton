from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge

from .models import Hypothesis


FEEDBACK_PATH = Path("data/feedback.json")


def apply_feedback_reranking(hypotheses: list[Hypothesis], path: Path = FEEDBACK_PATH) -> list[Hypothesis]:
    records = _load_feedback(path)
    if not records:
        return hypotheses
    model = _train_feedback_model(records)
    adjusted = [_apply_one(hypothesis, records, model) for hypothesis in hypotheses]
    return sorted(adjusted, key=lambda item: item.total_score, reverse=True)


def _apply_one(hypothesis: Hypothesis, records: list[dict], model: tuple[HashingVectorizer, Ridge] | None) -> Hypothesis:
    signal = 0.0
    matched = 0
    tags = set(hypothesis.tags)
    title_words = set(hypothesis.title.lower().split())
    for record in records:
        record_tags = set(record.get("tags", []))
        record_title_words = set(str(record.get("title", "")).lower().split())
        overlap = len(tags & record_tags) + len(title_words & record_title_words) * 0.3
        if overlap <= 0:
            continue
        matched += 1
        status = record.get("status")
        if status == "confirmed":
            signal += min(0.08, 0.025 * overlap)
        elif status == "rejected":
            signal -= min(0.10, 0.03 * overlap)
        elif status == "needs_validation":
            signal += min(0.03, 0.01 * overlap)
    ml_signal = _predict_feedback_signal(hypothesis, model)
    signal += ml_signal
    if matched == 0 and ml_signal == 0:
        return hypothesis
    total_score = min(1.0, max(0.0, hypothesis.total_score + signal))
    confidence = min(1.0, max(0.0, hypothesis.confidence + signal * 0.5))
    risk = min(1.0, max(0.0, hypothesis.risk - signal * 0.25))
    rationale = (
        f"{hypothesis.rationale} FeedbackAgent учел {matched} похожих экспертных оценок, "
        f"ML-модель по истории дала поправку {ml_signal:+.3f}, итоговый score изменен на {signal:+.3f}."
    )
    return replace(
        hypothesis,
        total_score=round(total_score, 3),
        confidence=round(confidence, 3),
        risk=round(risk, 3),
        rationale=rationale,
    )


def _load_feedback(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _train_feedback_model(records: list[dict]) -> tuple[HashingVectorizer, Ridge] | None:
    labeled = [record for record in records if record.get("status") in {"confirmed", "rejected", "needs_validation"}]
    if len(labeled) < 2:
        return None
    vectorizer = HashingVectorizer(
        lowercase=True,
        alternate_sign=False,
        n_features=2**12,
        norm="l2",
    )
    texts = [_record_text(record) for record in labeled]
    targets = [_status_target(record.get("status")) for record in labeled]
    model = Ridge(alpha=1.0)
    model.fit(vectorizer.transform(texts), targets)
    return vectorizer, model


def _predict_feedback_signal(hypothesis: Hypothesis, model: tuple[HashingVectorizer, Ridge] | None) -> float:
    if model is None:
        return 0.0
    vectorizer, regressor = model
    prediction = float(regressor.predict(vectorizer.transform([_hypothesis_text(hypothesis)]))[0])
    return max(-0.08, min(0.08, prediction * 0.08))


def _record_text(record: dict) -> str:
    return " ".join(
        [
            str(record.get("title", "")),
            str(record.get("statement", "")),
            " ".join(map(str, record.get("tags", []))),
        ]
    )


def _hypothesis_text(hypothesis: Hypothesis) -> str:
    return " ".join([hypothesis.title, hypothesis.statement, hypothesis.mechanism, " ".join(hypothesis.tags)])


def _status_target(status: str | None) -> float:
    if status == "confirmed":
        return 1.0
    if status == "rejected":
        return -1.0
    return 0.35
