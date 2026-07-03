from __future__ import annotations

import networkx as nx

from .models import Hypothesis


def build_relation_graph(hypotheses: list[Hypothesis]) -> nx.Graph:
    graph = nx.Graph()
    for hypothesis in hypotheses:
        hypothesis_node = hypothesis.title
        graph.add_node(hypothesis_node, kind="hypothesis", size=18)
        for tag in hypothesis.tags:
            graph.add_node(tag, kind="factor", size=12)
            graph.add_edge(hypothesis_node, tag, weight=hypothesis.total_score)
        for evidence in hypothesis.evidence[:2]:
            source = evidence.source
            graph.add_node(source, kind="source", size=9)
            graph.add_edge(hypothesis_node, source, weight=evidence.score)
    return graph


def graph_to_plotly_data(graph: nx.Graph) -> tuple[list[float], list[float], list[str], list[int], list[float], list[float]]:
    if not graph.nodes:
        return [], [], [], [], [], []
    positions = nx.spring_layout(graph, seed=42, k=0.8)
    node_x: list[float] = []
    node_y: list[float] = []
    labels: list[str] = []
    sizes: list[int] = []
    for node, attrs in graph.nodes(data=True):
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(str(node))
        sizes.append(int(attrs.get("size", 10)))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for start, finish in graph.edges():
        x0, y0 = positions[start]
        x1, y1 = positions[finish]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    return node_x, node_y, labels, sizes, edge_x, edge_y
