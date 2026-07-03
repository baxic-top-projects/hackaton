from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    source: str
    quote: str
    score: float


@dataclass(frozen=True)
class CalculationResult:
    name: str
    status: str
    value: str
    rationale: str
    assumptions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Hypothesis:
    title: str
    statement: str
    mechanism: str
    rationale: str
    novelty: float
    feasibility: float
    expected_value: float
    risk: float
    confidence: float
    total_score: float
    evidence: list[Evidence]
    experiment_plan: list[str]
    risks: list[str]
    resources: list[str]
    tags: list[str]
    calculations: list[CalculationResult] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchBrief:
    target: str
    constraints: str
    available_materials: str
    equipment: str
    budget: str
    weights: dict[str, float]
