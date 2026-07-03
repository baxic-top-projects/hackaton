from __future__ import annotations

import re
from dataclasses import dataclass

import networkx as nx

from .hypothesis_engine import ADDITIVES, PROCESSES, PROPERTIES
from .models import Chunk, Evidence
from .retrieval import KnowledgeIndex


ENTITY_TERMS = {
    **{term: "material" for term in ADDITIVES},
    **{term: "process" for term in PROCESSES},
    **{term: "property" for term in PROPERTIES},
}

PARAMETER_PATTERNS = {
    "temperature": r"\b\d{2,4}\s?(?:°C|C|℃)\b",
    "percent": r"\b\d+(?:[.,]\d+)?\s?%",
    "particle_size": r"(?:[+-]\s?\d+\s?(?:мкм|mm|мм)|\b\d+\s?(?:мкм|mm|мм)\b)",
    "ph": r"\bpH\s?\d+(?:[.,]\d+)?\b",
    "mass": r"\b\d+(?:[.,]\d+)?\s?(?:т|тонн|kg|кг)\b",
}

CHEMICAL_SYMBOLS = {
    "Ni": "material",
    "Cu": "material",
    "Co": "material",
    "Fe": "material",
    "S": "material",
    "Pt": "material",
    "Pd": "material",
    "Au": "material",
    "Ag": "material",
}


@dataclass(frozen=True)
class GraphContext:
    evidence: list[Evidence]
    related_terms: list[str]
    graph_path: list[str]


@dataclass
class GraphRAGIndex:
    graph: nx.Graph
    chunk_by_id: dict[str, Chunk]

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "GraphRAGIndex":
        graph = nx.Graph()
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        for chunk in chunks:
            chunk_node = f"chunk:{chunk.id}"
            source_node = f"source:{chunk.source}"
            graph.add_node(chunk_node, kind="chunk", label=chunk.id)
            graph.add_node(source_node, kind="source", label=chunk.source)
            graph.add_edge(chunk_node, source_node, relation="from_source", weight=1.0)

            terms = extract_terms(chunk.text)
            for term in terms:
                term_node = f"term:{term}"
                graph.add_node(term_node, kind=_term_kind(term), label=term)
                graph.add_edge(chunk_node, term_node, relation="mentions", weight=1.0)
            for left, right in _pairs(terms):
                graph.add_edge(f"term:{left}", f"term:{right}", relation="co_occurs", weight=0.6)
        return cls(graph=graph, chunk_by_id=chunk_by_id)

    def retrieve(self, query: str, text_index: KnowledgeIndex, limit: int = 6) -> GraphContext:
        seed = text_index.search(query, limit=limit)
        query_terms = extract_terms(query)
        related = self._related_terms(query_terms, max_terms=8)
        expanded_query = " ".join([query, *related])
        expanded = text_index.search(expanded_query, limit=limit)
        evidence = _merge_evidence(seed, expanded)
        path = self._explain_path(query_terms, related)
        return GraphContext(evidence=evidence[:limit], related_terms=related, graph_path=path)

    def stats(self) -> dict[str, int]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "materials": _count_kind(self.graph, "material"),
            "processes": _count_kind(self.graph, "process"),
            "properties": _count_kind(self.graph, "property"),
            "parameters": _count_kind(self.graph, "parameter"),
        }

    def _related_terms(self, terms: list[str], max_terms: int) -> list[str]:
        scores: dict[str, float] = {}
        for term in terms:
            node = f"term:{term}"
            if node not in self.graph:
                continue
            for neighbor in self.graph.neighbors(node):
                if not neighbor.startswith("term:"):
                    for second_hop in self.graph.neighbors(neighbor):
                        if second_hop.startswith("term:") and second_hop != node:
                            label = second_hop.removeprefix("term:")
                            scores[label] = scores.get(label, 0.0) + 0.7
                    continue
                label = neighbor.removeprefix("term:")
                scores[label] = scores.get(label, 0.0) + 1.0
        return [term for term, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_terms]]

    def _explain_path(self, query_terms: list[str], related_terms: list[str]) -> list[str]:
        paths: list[str] = []
        for start in query_terms[:4]:
            for finish in related_terms[:4]:
                if start == finish:
                    continue
                start_node = f"term:{start}"
                finish_node = f"term:{finish}"
                if start_node not in self.graph or finish_node not in self.graph:
                    continue
                try:
                    path = nx.shortest_path(self.graph, start_node, finish_node)
                except nx.NetworkXNoPath:
                    continue
                labels = [self.graph.nodes[node].get("label", node) for node in path]
                paths.append(" -> ".join(labels))
                if len(paths) >= 5:
                    return paths
        return paths


def extract_terms(text: str) -> list[str]:
    found = []
    text_lower = text.lower()
    for term in ENTITY_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text_lower, flags=re.IGNORECASE):
            found.append(term)
    for symbol in CHEMICAL_SYMBOLS:
        if re.search(rf"\b{re.escape(symbol)}\b", text, flags=re.IGNORECASE):
            found.append(symbol)
    for kind, pattern in PARAMETER_PATTERNS.items():
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            value = normalize_parameter(match)
            found.append(f"{kind}:{value}")
    return list(dict.fromkeys(found))


def normalize_parameter(value: str) -> str:
    return re.sub(r"\s+", "", value).replace(",", ".")


def _term_kind(term: str) -> str:
    if term in ENTITY_TERMS:
        return ENTITY_TERMS[term]
    if term in CHEMICAL_SYMBOLS:
        return CHEMICAL_SYMBOLS[term]
    if ":" in term:
        return "parameter"
    return "term"


def _pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for idx, left in enumerate(items):
        for right in items[idx + 1 :]:
            pairs.append((left, right))
    return pairs


def _merge_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    merged: dict[str, Evidence] = {}
    for item in [*left, *right]:
        current = merged.get(item.chunk_id)
        if current is None or item.score > current.score:
            merged[item.chunk_id] = item
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def _count_kind(graph: nx.Graph, kind: str) -> int:
    return sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("kind") == kind)
