from __future__ import annotations

import json
import os
import re
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from .graph_rag import GraphRAGIndex


DEFAULT_GRAPH_DIR = Path("data/graph_store")


def persist_graph(index: GraphRAGIndex, target_dir: Path = DEFAULT_GRAPH_DIR) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "knowledge_graph.json"
    graphml_path = target_dir / "knowledge_graph.graphml"
    turtle_path = target_dir / "knowledge_graph.ttl"

    payload = json_graph.node_link_data(index.graph, edges="edges")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    nx.write_graphml(index.graph, graphml_path)
    turtle_path.write_text(_to_turtle(index.graph), encoding="utf-8")
    neo4j_status = _push_to_neo4j(index.graph)

    return {
        "json": str(json_path),
        "graphml": str(graphml_path),
        "ttl": str(turtle_path),
        "neo4j": neo4j_status,
    }


def _to_turtle(graph: nx.Graph) -> str:
    lines = [
        "@prefix hf: <https://hypothesis-factory.local/ontology#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "",
    ]
    for node, attrs in graph.nodes(data=True):
        subject = _iri(node)
        label = _literal(attrs.get("label", node))
        kind = _literal(attrs.get("kind", "node"))
        lines.append(f"{subject} a hf:{_class_name(attrs.get('kind', 'Node'))} ;")
        lines.append(f"    rdfs:label \"{label}\" ;")
        lines.append(f"    hf:kind \"{kind}\" .")
    for left, right, attrs in graph.edges(data=True):
        relation = _predicate(attrs.get("relation", "relatedTo"))
        lines.append(f"{_iri(left)} hf:{relation} {_iri(right)} .")
    return "\n".join(lines) + "\n"


def _push_to_neo4j(graph: nx.Graph) -> str:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not (uri and user and password):
        return "not_configured"
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return "neo4j_driver_missing"
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            for node, attrs in graph.nodes(data=True):
                session.run(
                    """
                    MERGE (n:HFNode {id: $id})
                    SET n.label = $label, n.kind = $kind
                    """,
                    id=node,
                    label=attrs.get("label", node),
                    kind=attrs.get("kind", "node"),
                )
            for left, right, attrs in graph.edges(data=True):
                session.run(
                    """
                    MATCH (a:HFNode {id: $left})
                    MATCH (b:HFNode {id: $right})
                    MERGE (a)-[r:HF_RELATION {relation: $relation}]->(b)
                    SET r.weight = $weight
                    """,
                    left=left,
                    right=right,
                    relation=attrs.get("relation", "related"),
                    weight=float(attrs.get("weight", 1.0)),
                )
        driver.close()
    except Exception as exc:
        return f"error: {exc}"
    return "synced"


def _iri(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "node"
    return f"hf:{slug}"


def _predicate(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "relatedTo"


def _class_name(value: str) -> str:
    cleaned = _predicate(value)
    return cleaned[:1].upper() + cleaned[1:]


def _literal(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')
