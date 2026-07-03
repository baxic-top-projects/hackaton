from __future__ import annotations

from dataclasses import dataclass, replace

from .calculators import run_calculators
from .graph_rag import GraphContext, GraphRAGIndex
from .hypothesis_engine import generate_hypotheses
from .models import Chunk, Hypothesis, ResearchBrief
from .retrieval import KnowledgeIndex


@dataclass(frozen=True)
class AgentStep:
    agent: str
    status: str
    result: str


@dataclass(frozen=True)
class AgenticResult:
    hypotheses: list[Hypothesis]
    graph_index: GraphRAGIndex
    graph_context: GraphContext
    steps: list[AgentStep]
    long_context: str


def run_agentic_factory(
    brief: ResearchBrief,
    text_index: KnowledgeIndex,
    chunks: list[Chunk],
    limit: int,
) -> AgenticResult:
    steps = [
        AgentStep("IngestionAgent", "ok", f"Подготовлено фрагментов знаний: {len(chunks)}"),
    ]
    graph_index = GraphRAGIndex.build(chunks)
    graph_context = graph_index.retrieve(_brief_query(brief), text_index, limit=8)
    stats = graph_index.stats()
    steps.append(
        AgentStep(
            "GraphRAGAgent",
            "ok",
            (
                f"Граф: {stats['nodes']} узлов, {stats['edges']} связей; "
                f"связанные термины: {', '.join(graph_context.related_terms[:6]) or 'не найдены'}"
            ),
        )
    )

    hypotheses = generate_hypotheses(brief, text_index, limit=limit)
    enriched = [_enrich_hypothesis(item, brief, graph_context) for item in hypotheses]
    steps.append(AgentStep("HypothesisAgent", "ok", f"Сгенерировано гипотез: {len(enriched)}"))
    steps.append(
        AgentStep(
            "CalculatorAgent",
            "ok",
            f"Выполнено расчетных проверок: {sum(len(item.calculations) for item in enriched)}",
        )
    )
    long_context = _build_long_context(brief, enriched, graph_context)
    steps.append(
        AgentStep(
            "LongContextLLMAgent",
            "ready",
            f"Подготовлен контекст для YandexGPT: {len(long_context)} символов",
        )
    )
    return AgenticResult(
        hypotheses=enriched,
        graph_index=graph_index,
        graph_context=graph_context,
        steps=steps,
        long_context=long_context,
    )


def _brief_query(brief: ResearchBrief) -> str:
    return " ".join([brief.target, brief.constraints, brief.available_materials, brief.equipment, brief.budget])


def _enrich_hypothesis(
    hypothesis: Hypothesis,
    brief: ResearchBrief,
    graph_context: GraphContext,
) -> Hypothesis:
    graph_evidence = [
        item for item in graph_context.evidence if item.chunk_id not in {e.chunk_id for e in hypothesis.evidence}
    ][:2]
    evidence = [*hypothesis.evidence, *graph_evidence]
    graph_terms = ", ".join(graph_context.related_terms[:5]) or "нет дополнительных терминов"
    rationale = (
        f"{hypothesis.rationale} GraphRAG расширил контекст через связанные узлы: {graph_terms}."
    )
    calculations = run_calculators(hypothesis, brief)
    risks = [
        *hypothesis.risks,
        *[
            f"{result.name}: {result.value}"
            for result in calculations
            if result.status in {"watch", "risk"}
        ],
    ]
    resources = [
        *hypothesis.resources,
        "GraphRAG-карта связей источников",
        "расчетный лист калькуляторов",
    ]
    confidence = min(1.0, hypothesis.confidence + 0.04 * len(graph_evidence))
    total_score = min(1.0, hypothesis.total_score + 0.02 * len(calculations))
    return replace(
        hypothesis,
        rationale=rationale,
        evidence=evidence,
        risks=risks,
        resources=resources,
        confidence=round(confidence, 3),
        total_score=round(total_score, 3),
        calculations=calculations,
    )


def _build_long_context(
    brief: ResearchBrief,
    hypotheses: list[Hypothesis],
    graph_context: GraphContext,
) -> str:
    lines = [
        "BRIEF",
        f"Target: {brief.target}",
        f"Constraints: {brief.constraints}",
        f"Materials: {brief.available_materials}",
        f"Equipment: {brief.equipment}",
        "",
        "GRAPHRAG RELATED TERMS",
        ", ".join(graph_context.related_terms),
        "",
        "GRAPH PATHS",
        *graph_context.graph_path,
        "",
        "EVIDENCE",
    ]
    for evidence in graph_context.evidence:
        lines.append(f"- {evidence.source} ({evidence.score}): {evidence.quote}")
    lines.append("")
    lines.append("HYPOTHESES WITH CALCULATOR RESULTS")
    for idx, hypothesis in enumerate(hypotheses, start=1):
        lines.extend(
            [
                f"{idx}. {hypothesis.title}",
                hypothesis.statement,
                f"Mechanism: {hypothesis.mechanism}",
                f"Score: {hypothesis.total_score}",
            ]
        )
        for result in hypothesis.calculations:
            lines.append(f"Calculator: {result.name} | {result.status} | {result.value}")
    return "\n".join(lines)
