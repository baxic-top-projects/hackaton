from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Chunk, Evidence


@dataclass
class KnowledgeIndex:
    chunks: list[Chunk]
    vectorizer: TfidfVectorizer
    matrix: object

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "KnowledgeIndex":
        if not chunks:
            raise ValueError("At least one knowledge chunk is required.")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=12000,
            token_pattern=r"(?u)\b[\w.%+-]+\b",
        )
        matrix = vectorizer.fit_transform(chunk.text for chunk in chunks)
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
