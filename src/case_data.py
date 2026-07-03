from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .ingestion import load_documents_from_paths, normalize_text
from .models import Document


CASE_DATA_DIR = Path("data/case_yandex")
MAX_INDEXED_PDF_MB = 15


@dataclass(frozen=True)
class TailingsMetric:
    source: str
    stream: str
    smt: float | None
    ni_percent: float | None
    ni_tons: float | None
    cu_percent: float | None
    cu_tons: float | None


@dataclass(frozen=True)
class CaseProfile:
    target: str
    constraints: str
    available_materials: str
    equipment: str
    budget: str
    metrics: list[TailingsMetric] = field(default_factory=list)
    particle_classes: list[str] = field(default_factory=list)
    loss_forms: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)


def case_data_available(case_dir: Path = CASE_DATA_DIR) -> bool:
    return case_dir.exists() and any(case_dir.rglob("*"))


def build_case_profile(case_dir: Path = CASE_DATA_DIR) -> CaseProfile:
    metrics: list[TailingsMetric] = []
    particle_classes: set[str] = set()
    loss_forms: set[str] = set()
    source_files: list[str] = []

    for workbook in case_dir.rglob("*.xlsx"):
        source_files.append(str(workbook))
        frame = pd.read_excel(workbook, header=None).fillna("")
        metrics.extend(_extract_tailings_metrics(frame, workbook))
        particle_classes.update(_extract_particle_classes(frame))
        loss_forms.update(_extract_loss_forms(frame))

    for path in case_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".docx", ".pdf", ".png"}:
            source_files.append(str(path))

    total_tailings = sum(metric.smt or 0 for metric in metrics if "отвальные" in metric.stream.lower())
    total_ni = sum(metric.ni_tons or 0 for metric in metrics if "отвальные" in metric.stream.lower())
    total_cu = sum(metric.cu_tons or 0 for metric in metrics if "отвальные" in metric.stream.lower())
    classes = ", ".join(sorted(particle_classes, key=_particle_sort_key)[:8])
    forms = ", ".join(sorted(loss_forms)[:8])

    target = (
        "Снизить потери никеля и меди в отвальных хвостах флотационного обогащения "
        "за счет выбора проверяемых технологических гипотез"
    )
    constraints = (
        "Использовать данные кейса по хвостам обогатительных фабрик; "
        "фокус на элементах 28 и 29, трактуемых как Ni и Cu; "
        f"учесть суммарные отвальные хвосты около {total_tailings:,.0f} СМТ, "
        f"потери Ni около {total_ni:,.0f} т и Cu около {total_cu:,.0f} т; "
        f"учесть классы крупности: {classes or 'нет данных'}; "
        f"учесть формы потерь: {forms or 'нет данных'}; "
        "не предлагать гипотезы без лабораторной проверяемости, материального баланса и контроля извлечения."
    )
    available_materials = (
        "отвальные, породные и пирротиновые хвосты; шихта руд; материал пруда-накопителя; "
        "минеральные формы Pnt/Cp, пирротин, валлериит, пирит, миллерит"
    )
    equipment = (
        "флотационная схема, измельчение, классификация по крупности, анализ хвостов, "
        "лабораторные флотационные тесты, материальный баланс, регламентное оборудование из схем кейса"
    )
    budget = "приоритет быстрым лабораторным тестам на существующей схеме флотации и без капитальной перестройки"

    return CaseProfile(
        target=target,
        constraints=constraints,
        available_materials=available_materials,
        equipment=equipment,
        budget=budget,
        metrics=metrics,
        particle_classes=sorted(particle_classes, key=_particle_sort_key),
        loss_forms=sorted(loss_forms),
        source_files=sorted(set(source_files)),
    )


def load_case_documents(case_dir: Path = CASE_DATA_DIR) -> list[Document]:
    indexable_paths = []
    metadata_docs: list[Document] = []
    for path in case_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".docx", ".xlsx", ".txt", ".md"}:
            indexable_paths.append(path)
            continue
        if suffix == ".pdf" and path.stat().st_size <= MAX_INDEXED_PDF_MB * 1024 * 1024:
            indexable_paths.append(path)
            continue
        if suffix in {".pdf", ".png"}:
            metadata_docs.append(_metadata_document(path))
    return [*load_documents_from_paths(indexable_paths), *metadata_docs, _profile_document(build_case_profile(case_dir))]


def _extract_tailings_metrics(frame: pd.DataFrame, workbook: Path) -> list[TailingsMetric]:
    metrics: list[TailingsMetric] = []
    for _, row in frame.iterrows():
        values = row.tolist()
        start_idx = _first_non_empty_index(values)
        if start_idx is None:
            continue
        stream = str(values[start_idx]).strip()
        if not stream or not any(word in stream.lower() for word in ["хвост", "шихта", "итого"]):
            continue
        if len(values) < start_idx + 6:
            continue
        metric = TailingsMetric(
            source=workbook.name,
            stream=stream,
            smt=_to_float(values[start_idx + 1]),
            ni_percent=_to_float(values[start_idx + 2]),
            ni_tons=_to_float(values[start_idx + 3]),
            cu_percent=_to_float(values[start_idx + 4]),
            cu_tons=_to_float(values[start_idx + 5]),
        )
        if metric.smt or metric.ni_tons or metric.cu_tons:
            metrics.append(metric)
    return metrics


def _extract_particle_classes(frame: pd.DataFrame) -> set[str]:
    classes = set()
    for value in _iter_cells(frame):
        if any(marker in value for marker in ["+125", "-125", "-71", "-45", "-20", "-10"]):
            classes.add(value.replace("  ", " "))
    return classes


def _extract_loss_forms(frame: pd.DataFrame) -> set[str]:
    forms = set()
    keywords = ["Pnt/Cp", "пирротин", "Силикат", "Валлер", "Пирит", "Миллерит", "слот"]
    for text in _iter_cells(frame):
        if any(keyword.lower() in text.lower() for keyword in keywords):
            forms.add(text)
    return forms


def _metadata_document(path: Path) -> Document:
    parent = path.parent.name.lower()
    if "схем" in parent:
        text = f"Схема флотации из данных кейса: {path.name}. Учитывать как ограничение существующей технологической схемы."
    elif "регламент" in parent:
        text = f"Регламентный материал из данных кейса: {path.name}. Учитывать как ограничение по оборудованию и процедурам."
    else:
        text = f"Дополнительный источник данных кейса: {path.name}. Размер файла {path.stat().st_size} байт."
    return Document(source=path.name, text=normalize_text(text), metadata={"path": str(path), "extension": path.suffix})


def _profile_document(profile: CaseProfile) -> Document:
    lines = [
        "Сводка ограничений по данным кейса Яндекс.Диска.",
        f"Цель: {profile.target}",
        f"Ограничения: {profile.constraints}",
        f"Доступные материалы: {profile.available_materials}",
        f"Оборудование и процессы: {profile.equipment}",
        f"Бюджет/приоритет: {profile.budget}",
    ]
    for metric in profile.metrics[:20]:
        lines.append(
            f"{metric.source}: {metric.stream}, СМТ={metric.smt}, Ni={metric.ni_percent}%/{metric.ni_tons} т, "
            f"Cu={metric.cu_percent}%/{metric.cu_tons} т."
        )
    return Document(source="case_constraints_summary", text=normalize_text("\n".join(lines)))


def _to_float(value: object) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_non_empty_index(values: list[object]) -> int | None:
    for idx, value in enumerate(values):
        if str(value).strip():
            return idx
    return None


def _iter_cells(frame: pd.DataFrame) -> list[str]:
    values = []
    for row in frame.itertuples(index=False):
        for value in row:
            text = str(value).strip()
            if text:
                values.append(text)
    return values


def _particle_sort_key(value: str) -> tuple[int, str]:
    order = ["+125", "-125", "+71", "-71", "+45", "-45", "+20", "-20", "+10", "-10"]
    for idx, marker in enumerate(order):
        if marker in value:
            return idx, value
    return 99, value
