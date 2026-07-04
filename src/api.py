from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .agentic_pipeline import run_agentic_factory
from .auth import has_role, role_for_api_token
from .ingestion import chunk_documents, load_documents_from_paths, normalize_text
from .metadata import extract_metadata
from .models import Document, ResearchBrief
from .retrieval import KnowledgeIndex


SAMPLE_DIR = Path("data/sample_knowledge")

app = FastAPI(title="Hypothesis Factory API", version="1.0.0")


class ApiDocument(BaseModel):
    source: str = Field(default="api-document.txt", description="Название источника: файл, отчет, статья, таблица или патент.")
    text: str = Field(
        description=(
            "Извлеченный текст документа. Передавайте сюда содержимое отчета, статьи, таблицы, "
            "OCR-результат изображения или краткое описание источника."
        )
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Необязательные метаданные: дата, автор, тип источника, условия эксперимента.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "source": "flotation_report.txt",
                "text": (
                    "Отчет по хвостам флотации: повышенные потери Ni/Cu наблюдаются в классе "
                    "+71 мкм. Для проверки предлагаются доизмельчение, контроль pH 8.5-9.5 "
                    "и корректировка реагентного режима."
                ),
                "metadata": {"date": "2024", "type": "lab_report"},
            }
        }
    }


class GenerateRequest(BaseModel):
    target: str = Field(description="Цель или технологическая проблема, под которую нужно сгенерировать гипотезы.")
    constraints: str = Field(default="", description="Ограничения: сроки, бюджет, нормативы, доступные режимы, запреты.")
    available_materials: str = Field(default="", description="Доступное сырье, материалы, реагенты или потоки.")
    equipment: str = Field(default="", description="Доступное оборудование для проверки гипотез.")
    budget: str = Field(default="", description="Бюджетные и временные ограничения.")
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "novelty": 0.2,
            "feasibility": 0.25,
            "expected_value": 0.3,
            "risk": 0.15,
            "confidence": 0.1,
        }
    )
    limit: int = Field(default=6, ge=1, le=12)
    documents: list[ApiDocument] = Field(
        default_factory=list,
        description=(
            "Переданные документы в текстовом виде. Если список пустой, API использует встроенную demo-базу знаний."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "target": "снизить потери Ni/Cu в хвостах флотации",
                "constraints": "лабораторная проверка до 2 недель; без капитальных изменений схемы",
                "available_materials": "хвосты флотации, известь, собиратель, вспениватель",
                "equipment": "флотомашина, мельница, классификатор, pH-метр",
                "budget": "пилотный лабораторный скрининг",
                "limit": 3,
                "documents": [
                    {
                        "source": "flotation_report.txt",
                        "text": (
                            "Отчет по хвостам флотации: повышенные потери Ni/Cu наблюдаются "
                            "в классе +71 мкм. В пробах указаны pH 8.5-9.5, реагентный режим "
                            "и необходимость проверки доизмельчения перед перечисткой."
                        ),
                        "metadata": {"type": "lab_report", "year": "2024"},
                    },
                    {
                        "source": "scheme_ocr.txt",
                        "text": (
                            "OCR схемы: основная флотация, перечистка, хвостовой поток, "
                            "контроль крупности и pH."
                        ),
                        "metadata": {"type": "image_ocr"},
                    },
                ],
            }
        }
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def require_api_role(x_api_key: str | None = Header(default=None)) -> str:
    role = role_for_api_token(x_api_key)
    if role is None:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    if not has_role(role, "researcher"):
        raise HTTPException(status_code=403, detail="Insufficient API role.")
    return role


@app.post("/api/generate")
def generate(request: GenerateRequest, role: str = Depends(require_api_role)) -> dict[str, Any]:
    brief = ResearchBrief(
        target=request.target,
        constraints=request.constraints,
        available_materials=request.available_materials,
        equipment=request.equipment,
        budget=request.budget,
        weights=_normalize_weights(request.weights),
    )
    documents = _api_documents(request.documents)
    if not documents:
        documents = load_documents_from_paths(SAMPLE_DIR.glob("*"))
    chunks = chunk_documents(documents)
    if not chunks:
        raise HTTPException(status_code=400, detail="No indexable knowledge chunks were provided.")
    index = KnowledgeIndex.build(chunks)
    result = run_agentic_factory(brief, index, chunks, limit=request.limit)
    return {
        "brief": asdict(brief),
        "hypotheses": [asdict(hypothesis) for hypothesis in result.hypotheses],
        "steps": [asdict(step) for step in result.steps],
        "graph_stats": result.graph_index.stats(),
        "graph_paths": result.graph_files,
        "api_role": role,
    }


def _api_documents(documents: list[ApiDocument]) -> list[Document]:
    return [
        Document(
            source=document.source,
            text=normalize_text(document.text),
            metadata=extract_metadata(document.text, document.source, {**document.metadata, "origin": "api"}),
        )
        for document in documents
        if document.text.strip()
    ]


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in weights.values()) or 1.0
    return {key: max(value, 0.0) / total for key, value in weights.items()}
