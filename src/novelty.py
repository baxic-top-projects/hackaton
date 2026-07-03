from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote_plus

import requests

from .models import CalculationResult, Hypothesis


CACHE_PATH = Path("data/novelty_cache.json")


def novelty_check_enabled() -> bool:
    return os.getenv("NOVELTY_CHECK_ENABLED", "1").lower() not in {"0", "false", "no"}


def apply_external_novelty_check(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    if not novelty_check_enabled():
        return hypotheses
    cache = _load_cache()
    enriched = [_check_one(hypothesis, cache) for hypothesis in hypotheses]
    _save_cache(cache)
    return enriched


def _check_one(hypothesis: Hypothesis, cache: dict[str, dict]) -> Hypothesis:
    query = _query_for(hypothesis)
    if query not in cache:
        cache[query] = _search_external_sources(query)
    result = cache[query]
    overlap_count = (
        int(result.get("publication_count", 0))
        + int(result.get("semantic_scholar_count", 0))
        + int(result.get("patent_count", 0))
    )
    penalty = min(0.18, overlap_count * 0.015)
    novelty = max(0.05, hypothesis.novelty - penalty)
    status = "ok" if overlap_count <= 2 else "watch"
    value = (
        f"Crossref: {result.get('publication_count', 0)}, "
        f"Semantic Scholar: {result.get('semantic_scholar_count', 0)}, "
        f"патенты: {result.get('patent_count', 0)}"
    )
    rationale = (
        "Внешняя проверка ищет похожие формулировки в Crossref, Semantic Scholar и PatentsView. "
        "Большое число совпадений снижает novelty-score, но не удаляет гипотезу: эксперт должен проверить контекст."
    )
    calculation = CalculationResult(
        name="Внешняя проверка новизны",
        status=status,
        value=value,
        rationale=rationale,
        assumptions=result.get("examples", [])[:3] or ["проверка могла быть ограничена доступностью внешних API"],
    )
    total_score = max(0.0, hypothesis.total_score - penalty * 0.4)
    return replace(
        hypothesis,
        novelty=round(novelty, 3),
        total_score=round(total_score, 3),
        calculations=[*hypothesis.calculations, calculation],
    )


def _query_for(hypothesis: Hypothesis) -> str:
    tags = " ".join(hypothesis.tags[:4])
    return f"{hypothesis.title} {tags}".strip()


def _search_external_sources(query: str) -> dict:
    publications = _search_crossref(query)
    semantic_scholar = _search_semantic_scholar(query)
    patents = _search_patentsview(query)
    return {
        "publication_count": publications["count"],
        "semantic_scholar_count": semantic_scholar["count"],
        "patent_count": patents["count"],
        "examples": [*publications["examples"], *semantic_scholar["examples"], *patents["examples"]][:7],
    }


def _search_crossref(query: str) -> dict:
    try:
        response = requests.get(
            f"https://api.crossref.org/works?query={quote_plus(query)}&rows=3",
            timeout=4,
        )
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
    except Exception:
        return {"count": 0, "examples": ["Crossref недоступен"]}
    examples = []
    for item in items:
        title = " ".join(item.get("title", [])[:1]).strip()
        if title:
            examples.append(f"Crossref: {title}")
    return {"count": len(items), "examples": examples}


def _search_semantic_scholar(query: str) -> dict:
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": 3, "fields": "title,year,url"},
            timeout=4,
        )
        response.raise_for_status()
        papers = response.json().get("data", [])
    except Exception:
        return {"count": 0, "examples": ["Semantic Scholar недоступен"]}
    examples = []
    for paper in papers:
        title = paper.get("title")
        year = paper.get("year")
        if title:
            examples.append(f"Semantic Scholar {year or ''}: {title}".strip())
    return {"count": len(papers), "examples": examples}


def _search_patentsview(query: str) -> dict:
    payload = {
        "q": {"_text_any": {"patent_title": query}},
        "f": ["patent_title", "patent_number"],
        "o": {"per_page": 3},
    }
    try:
        response = requests.post("https://api.patentsview.org/patents/query", json=payload, timeout=4)
        response.raise_for_status()
        patents = response.json().get("patents", [])
    except Exception:
        return {"count": 0, "examples": ["PatentsView недоступен"]}
    examples = []
    for patent in patents:
        title = patent.get("patent_title")
        number = patent.get("patent_number")
        if title:
            examples.append(f"Patent {number}: {title}")
    return {"count": len(patents), "examples": examples}


def _load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
