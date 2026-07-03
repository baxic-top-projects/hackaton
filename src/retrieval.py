from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Chunk, Evidence


TOKEN_PATTERN = r"[\w.%+-]+"
@dataclass
class KnowledgeIndex:
    chunks: list[Chunk]
    vectorizer: HashingVectorizer
    matrix: object

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "KnowledgeIndex":
        original_chunks = chunks
        chunks = [chunk for chunk in chunks if _has_indexable_text(chunk.text)]
        if not chunks and original_chunks:
            chunks = _fallback_chunks(original_chunks)
        if not chunks:
            raise ValueError("At least one knowledge chunk is required.")
        texts = [chunk.text for chunk in chunks]
        vectorizer = HashingVectorizer(
            lowercase=True,
            alternate_sign=False,
            n_features=2**14,
            norm="l2",
            token_pattern=rf"(?u)\b{TOKEN_PATTERN}\b",
        )
        matrix = vectorizer.transform(texts)
        if matrix.nnz == 0:
            chunks = _fallback_chunks(chunks)
            matrix = vectorizer.transform([chunk.text for chunk in chunks])
        return cls(chunks=chunks, vectorizer=vectorizer, matrix=matrix)

    def search(self, query: str, limit: int = 5) -> list[Evidence]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = similarities.argsort()[::-1][:limit]
        evidence: list[Evidence] = []
        for index in ranked:
            score = float(similarities[index])
            if score <= 0:
                continue
            chunk = self.chunks[int(index)]
            evidence.append(
                Evidence(
                    chunk_id=chunk.id,
                    source=chunk.source,
                    quote=_short_quote(chunk.text),
                    score=round(score, 3),
                )
            )
        return evidence


def _short_quote(text: str, max_chars: int = 520) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def _has_indexable_text(text: str) -> bool:
    return bool(re.search(r"[\wа-яА-ЯёЁ]", text or ""))


def _fallback_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [
        Chunk(
            id=chunk.id,
            source=chunk.source,
            text=f"source {chunk.source} document visual schema regulation uploaded file",
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]
