from __future__ import annotations

from dataclasses import dataclass, replace

from .calculators import run_calculators
from .counterfactual import apply_counterfactual_analysis
from .feedback import apply_feedback_reranking
from .graph_rag import GraphContext, GraphRAGIndex
from .graph_storage import persist_graph, query_neo4j_graph_stats
from .hypothesis_engine import generate_hypotheses
from .models import Chunk, Hypothesis, ResearchBrief
from .novelty import apply_external_novelty_check, novelty_check_enabled
from .predictive import apply_predictive_kpi_model
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
    graph_files: dict[str, str]


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
    graph_files = persist_graph(graph_index)
    neo4j_stats = query_neo4j_graph_stats()
    graph_context = graph_index.retrieve(_brief_query(brief), text_index, limit=8)
    stats = graph_index.stats()
    steps.append(
        AgentStep(
            "GraphRAGAgent",
            "ok",
            (
                f"Граф: {stats['nodes']} узлов, {stats['edges']} связей; "
                f"связанные термины: {', '.join(graph_context.related_terms[:6]) or 'не найдены'}; "
                f"сохранен: {graph_files.get('json')}"
            ),
        )
    )
    if neo4j_stats.get("status") == "ok":
        steps.append(
            AgentStep(
                "Neo4jGraphAgent",
                "ok",
                f"Neo4j queried: {neo4j_stats.get('nodes')} узлов, {neo4j_stats.get('edges')} связей.",
            )
        )

    hypotheses = generate_hypotheses(brief, text_index, limit=limit)
    enriched = [_enrich_hypothesis(item, brief, graph_context) for item in hypotheses]
    enriched = apply_counterfactual_analysis(enriched, brief)
    steps.append(AgentStep("CounterfactualAgent", "ok", "Сравнены baseline, сценарий без фактора и альтернативный процесс."))
    enriched = apply_predictive_kpi_model(enriched, brief)
    steps.append(AgentStep("PredictiveKPIAgent", "ok", "RandomForest surrogate model пересчитал expected KPI uplift."))
    if novelty_check_enabled():
        enriched = apply_external_novelty_check(enriched)
        steps.append(AgentStep("NoveltyAgent", "ok", "Проверка похожих публикаций и патентов выполнена."))
    reranked = apply_feedback_reranking(enriched)
    if reranked != enriched:
        steps.append(AgentStep("FeedbackAgent", "ok", "Ранжирование скорректировано по экспертной истории."))
    enriched = reranked
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
        graph_files=graph_files,
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
    enriched = replace(
        hypothesis,
        rationale=rationale,
        evidence=evidence,
        risks=risks,
        resources=resources,
        confidence=round(confidence, 3),
        total_score=round(total_score, 3),
    )
    object.__setattr__(enriched, "calculations", calculations)
    return enriched


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
