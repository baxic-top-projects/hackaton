from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from streamlit.runtime.scriptrunner import get_script_run_ctx


logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)


def _launch_streamlit() -> int:
    env = os.environ.copy()
    env["HYPOTHESIS_FACTORY_STREAMLIT"] = "1"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.headless=true",
        "--server.port=8502",
        "--browser.gatherUsageStats=false",
        *sys.argv[1:],
    ]
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__" and get_script_run_ctx() is None and os.getenv("HYPOTHESIS_FACTORY_STREAMLIT") != "1":
    raise SystemExit(_launch_streamlit())


import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.agentic_pipeline import AgenticResult, run_agentic_factory
from src.auth import has_role, role_for_ui_token
from src.exporters import (
    hypotheses_to_docx,
    hypotheses_to_frame,
    hypotheses_to_json,
    hypotheses_to_markdown,
    hypotheses_to_pdf,
)
from src.graphing import build_relation_graph, graph_to_plotly_data
from src.ingestion import chunk_documents, load_documents_from_paths, load_uploaded_files
from src.models import ResearchBrief
from src.retrieval import KnowledgeIndex
from src.trackers import configured_trackers, create_jira_issue, create_youtrack_issue
from src.yandex_ai import generate_expert_summary, is_yandex_configured


SAMPLE_DIR = Path("data/sample_knowledge")
FEEDBACK_PATH = Path("data/feedback.json")


load_dotenv()


def _check_ui_access() -> bool:
    provided = st.sidebar.text_input("Access token", type="password")
    role = role_for_ui_token(provided)
    if role is not None:
        st.session_state["authenticated_role"] = role
        st.session_state.setdefault("role", role)
        return True
    st.warning("Введите access token для локального контура НИИ.")
    return False


def _render_role_selector() -> None:
    role = st.selectbox(
        "Роль",
        ["researcher", "expert", "viewer", "admin"],
        index=["researcher", "expert", "viewer", "admin"].index(st.session_state.get("role", st.session_state.get("authenticated_role", "researcher"))),
        help="viewer только смотрит отчеты; expert/admin могут сохранять feedback и создавать задачи.",
    )
    authenticated_role = st.session_state.get("authenticated_role", "viewer")
    if has_role(authenticated_role, role):
        st.session_state["role"] = role
    else:
        st.session_state["role"] = authenticated_role
        st.caption(f"Роль ограничена токеном: {authenticated_role}")


def _can_edit() -> bool:
    return has_role(st.session_state.get("role"), "expert")


def main() -> None:
    st.set_page_config(page_title="Фабрика гипотез", page_icon="H", layout="wide")
    st.title("Фабрика гипотез")
    st.caption("Гибридная агентная архитектура: GraphRAG + Long-Context LLM + расчетные калькуляторы")
    if not _check_ui_access():
        return

    use_yandex = is_yandex_configured()

    with st.sidebar:
        _render_role_selector()
        st.header("Входные данные")
        target = st.text_area(
            "Целевое свойство или технологическая проблема",
            value="Повысить жаропрочность никелевого сплава на 15% без существенного снижения пластичности",
            height=90,
        )
        constraints = st.text_area(
            "Ограничения",
            value="Использовать доступные легирующие элементы; избегать сильного роста себестоимости; лабораторная проверка за 2 недели",
            height=140,
        )
        available_materials = st.text_input(
            "Доступное сырье",
            value="никель, хром, ниобий, титан, молибден, бор",
        )
        equipment = st.text_input(
            "Оборудование",
            value="вакуумная плавка, печь отжига, установка старения, испытательная машина",
        )
        budget = st.text_input(
            "Бюджет/сроки",
            value="средний бюджет, 2 недели на первичный скрининг",
        )

        st.subheader("Веса ранжирования")
        novelty = st.slider("Новизна", 0.0, 1.0, 0.20, 0.05)
        feasibility = st.slider("Реализуемость", 0.0, 1.0, 0.25, 0.05)
        expected_value = st.slider("Ожидаемая ценность", 0.0, 1.0, 0.30, 0.05)
        risk = st.slider("Низкий риск", 0.0, 1.0, 0.15, 0.05)
        confidence = st.slider("Уверенность источников", 0.0, 1.0, 0.10, 0.05)

        uploaded_files = st.file_uploader(
            "Загрузить базу знаний",
            type=["txt", "md", "pdf", "docx", "csv", "xlsx", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        limit = st.slider("Количество гипотез", 3, 12, 6)
        if use_yandex:
            st.caption("YandexGPT используется автоматически для экспертной сводки.")
        else:
            st.caption("Для LLM-режима задайте YANDEX_API_KEY и YANDEX_FOLDER_ID.")
        run = st.button("Сгенерировать гипотезы", type="primary", use_container_width=True)

    weights = _normalize_weights(
        {
            "novelty": novelty,
            "feasibility": feasibility,
            "expected_value": expected_value,
            "risk": risk,
            "confidence": confidence,
        }
    )
    brief = ResearchBrief(
        target=target,
        constraints=constraints,
        available_materials=available_materials,
        equipment=equipment,
        budget=budget,
        weights=weights,
    )

    if run:
        _generate_and_store_result(brief, uploaded_files, limit, use_yandex)

    saved = st.session_state.get("last_result")
    if not saved:
        st.info("Нажмите кнопку генерации. Если файлы не загружены, будет использована встроенная демо-база знаний.")
        _render_requirements_match()
        return

    _render_saved_result(saved)


def _generate_and_store_result(
    brief: ResearchBrief,
    uploaded_files,
    limit: int,
    use_yandex: bool,
) -> None:
    with st.spinner("Запускаю агентный пайплайн: GraphRAG, генерация, калькуляторы, LLM-контекст..."):
        if uploaded_files:
            documents = load_uploaded_files(uploaded_files)
        else:
            documents = load_documents_from_paths(SAMPLE_DIR.glob("*"))
        chunks = chunk_documents(documents)
        if not chunks:
            st.error("Не удалось извлечь текст из источников. Загрузите файлы или добавьте демо-данные.")
            return
        index = KnowledgeIndex.build(chunks)
        result = run_agentic_factory(brief, index, chunks, limit=limit)
        hypotheses = result.hypotheses

    yandex_summary = None
    yandex_error = None
    if use_yandex:
        try:
            with st.spinner("Передаю GraphRAG-контекст и расчеты в Yandex AI Studio..."):
                yandex_summary = generate_expert_summary(brief, hypotheses, long_context=result.long_context)
        except Exception as exc:
            yandex_error = f"YandexGPT недоступен, базовая генерация сохранена: {exc}"

    st.session_state["last_result"] = {
        "brief": brief,
        "result": result,
        "hypotheses": hypotheses,
        "document_count": len(documents),
        "source_types": _source_type_counts(documents),
        "source_inventory": _source_inventory(documents),
        "chunk_count": len(chunks),
        "yandex_summary": yandex_summary,
        "yandex_error": yandex_error,
    }


def _render_saved_result(saved: dict) -> None:
    brief = saved["brief"]
    result = saved["result"]
    hypotheses = saved["hypotheses"]
    st.success(
        f"Сформировано гипотез: {len(hypotheses)}. "
        f"Источников: {saved['document_count']}. Фрагментов: {saved['chunk_count']}."
    )
    _render_source_types(saved.get("source_types", {}), saved.get("source_inventory", []))
    _render_case_requirements_match()
    _render_yandex_summary(saved.get("yandex_summary"), saved.get("yandex_error"))
    _render_agent_trace(result)
    _render_graph_context(result)
    _render_all_sources(saved.get("source_inventory", []))
    _render_rank_table(hypotheses)
    _render_graph(hypotheses)
    _render_cards(hypotheses)
    _render_exports(hypotheses, brief)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def _source_type_counts(documents) -> dict[str, int]:
    counts = Counter()
    for document in documents:
        extension = document.metadata.get("extension", "unknown")
        counts[str(extension).lower().lstrip(".") or "unknown"] += 1
    return dict(sorted(counts.items()))


def _source_inventory(documents) -> list[dict]:
    inventory = []
    for document in documents:
        extension = str(document.metadata.get("extension", "")).lower().lstrip(".")
        if not extension:
            extension = Path(document.source).suffix.lower().lstrip(".") or "unknown"
        inventory.append({"source": document.source, "extension": extension, "metadata": document.metadata})
    return sorted(inventory, key=lambda item: (item["extension"], item["source"]))


def _render_source_types(source_types: dict[str, int], source_inventory: list[dict[str, str]]) -> None:
    if not source_types:
        return
    image_count = sum(source_types.get(ext, 0) for ext in ["png", "jpg", "jpeg", "webp"])
    summary = ", ".join(f"{ext}: {count}" for ext, count in source_types.items())
    if image_count:
        st.caption(f"Типы источников: {summary}. Изображений использовано: {image_count}.")
        image_sources = [
            item["source"]
            for item in source_inventory
            if item["extension"] in {"png", "jpg", "jpeg", "webp"}
        ]
        with st.expander("Использованные изображения"):
            for source in image_sources:
                st.write(f"- {source}")
    else:
        st.caption(f"Типы источников: {summary}.")


def _render_all_sources(source_inventory: list[dict[str, str]]) -> None:
    if not source_inventory:
        return
    st.subheader("Все загруженные источники")
    with st.expander(f"Показать все источники ({len(source_inventory)})", expanded=False):
        for item in source_inventory:
            metadata = item.get("metadata") or {}
            dates = ", ".join(metadata.get("dates", [])[:2])
            authors = ", ".join(metadata.get("authors", [])[:2])
            details = " · ".join(value for value in [f"даты: {dates}" if dates else "", f"авторы: {authors}" if authors else ""] if value)
            st.write(f"- {item['source']} · тип: {item['extension']}" + (f" · {details}" if details else ""))


def _render_requirements_match() -> None:
    st.subheader("Гибридная архитектура")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GraphRAG", "сущности + связи")
    col2.metric("LLM", "long-context")
    col3.metric("Калькуляторы", "правила + физика")
    col4.metric("Интеграция", "API + trackers")
    st.write(
        "Система принимает цель, ограничения и базу знаний, строит граф материалов, процессов, свойств и источников, "
        "расширяет поиск через GraphRAG, проверяет гипотезы расчетными агентами и передает полный контекст в YandexGPT."
    )


def _render_case_requirements_match() -> None:
    st.subheader("Соответствие требованиям кейса")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Интерпретируемость", "источники + GraphRAG")
    c2.metric("Проверяемость", "план эксперимента")
    c3.metric("Экспертная настройка", "веса + feedback")
    c4.metric("Экспорт/API", "CSV/JSON/PDF/API")
    with st.expander("Что закрыто"):
        st.write("- Конкретные проверяемые гипотезы с механизмом влияния.")
        st.write("- Обоснование через цитаты источников и связи графа.")
        st.write("- Ранжирование по новизне, реализуемости, эффекту, риску и уверенности.")
        st.write("- Расчетные проверки и прогнозный диапазон KPI-эффекта.")
        st.write("- Экспертная обратная связь влияет на следующие ранжирования.")
        st.write("- HTTP API `/api/generate`, экспорт графа GraphML/JSON, Jira/YouTrack при наличии токенов.")


def _render_rank_table(hypotheses) -> None:
    st.subheader("Ранжирование")
    frame = hypotheses_to_frame(hypotheses)
    st.dataframe(
        frame[
            [
                "rank",
                "title",
                "total_score",
                "novelty",
                "feasibility",
                "expected_value",
                "risk",
                "confidence",
                "sources",
                "calculators",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


def _render_yandex_summary(summary: str | None, error: str | None) -> None:
    if not summary and not error:
        return
    st.subheader("Long-Context LLM агент")
    if summary:
        st.info(summary)
    if error:
        st.warning(error)


def _render_agent_trace(result: AgenticResult) -> None:
    st.subheader("Трассировка агентов")
    cols = st.columns(len(result.steps))
    for col, step in zip(cols, result.steps):
        col.metric(step.agent, step.status)
        col.caption(step.result)


def _render_graph_context(result: AgenticResult) -> None:
    st.subheader("GraphRAG контекст")
    stats = result.graph_index.stats()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Узлы", stats["nodes"])
    c2.metric("Связи", stats["edges"])
    c3.metric("Материалы", stats["materials"])
    c4.metric("Процессы", stats["processes"])
    c5.metric("Свойства", stats["properties"])
    c6.metric("Параметры", stats.get("parameters", 0))
    if result.graph_context.related_terms:
        st.caption("Связанные узлы: " + ", ".join(result.graph_context.related_terms[:10]))
    if result.graph_context.graph_path:
        with st.expander("Объяснимые пути в графе"):
            for path in result.graph_context.graph_path:
                st.write(f"- {path}")
    graph_files = getattr(result, "graph_files", {})
    if graph_files:
        with st.expander("Персистентный граф знаний"):
            for kind, path in graph_files.items():
                graph_path = Path(path)
                if graph_path.exists():
                    st.download_button(
                        f"Скачать {kind.upper()}",
                        data=graph_path.read_bytes(),
                        file_name=graph_path.name,
                        mime="application/json" if kind == "json" else "application/xml",
                        key=f"graph_download_{kind}",
                    )


def _render_graph(hypotheses) -> None:
    st.subheader("Граф связей")
    graph = build_relation_graph(hypotheses)
    node_x, node_y, labels, sizes, edge_x, edge_y = graph_to_plotly_data(graph)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1},
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker={"size": [size * 2 for size in sizes]},
            hovertext=labels,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, use_container_width=True)


def _render_cards(hypotheses) -> None:
    st.subheader("Карточки гипотез")
    for idx, hypothesis in enumerate(hypotheses, start=1):
        with st.expander(f"{idx}. {hypothesis.title} · score {hypothesis.total_score:.3f}", expanded=idx <= 2):
            st.write(hypothesis.statement)
            st.markdown(f"**Механизм:** {hypothesis.mechanism}.")
            st.markdown(f"**Обоснование:** {hypothesis.rationale}")
            st.caption(_novelty_explanation(hypothesis))

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Новизна", f"{hypothesis.novelty:.2f}")
            c2.metric("Реализуемость", f"{hypothesis.feasibility:.2f}")
            c3.metric("Ценность", f"{hypothesis.expected_value:.2f}")
            c4.metric("Риск", f"{hypothesis.risk:.2f}")
            c5.metric("Уверенность", f"{hypothesis.confidence:.2f}")

            st.markdown("**Дорожная карта проверки**")
            for step in hypothesis.experiment_plan:
                st.write(f"- {step}")

            st.markdown("**Риски**")
            for risk in hypothesis.risks:
                st.write(f"- {risk}")

            if hypothesis.calculations:
                st.markdown("**Расчетные проверки**")
                for result in hypothesis.calculations:
                    st.write(f"- {result.name}: `{result.status}` · {result.value}")
                    st.caption(result.rationale)

            st.markdown("**Источники**")
            for item in hypothesis.evidence:
                source_type = Path(item.source).suffix.lower().lstrip(".") or "unknown"
                st.caption(f"{item.source} · тип: {source_type} · релевантность {item.score:.3f}")
                st.write(item.quote)

            _render_feedback_controls(idx, hypothesis)
            _render_tracker_controls(idx, hypothesis)


def _render_exports(hypotheses, brief: ResearchBrief) -> None:
    st.subheader("Экспорт")
    frame = hypotheses_to_frame(hypotheses)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.download_button(
        "CSV для таск-трекера",
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="hypotheses.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col2.download_button(
        "JSON",
        data=hypotheses_to_json(hypotheses, brief).encode("utf-8"),
        file_name="hypotheses.json",
        mime="application/json",
        use_container_width=True,
    )
    col3.download_button(
        "Markdown-отчет",
        data=hypotheses_to_markdown(hypotheses, brief).encode("utf-8"),
        file_name="hypotheses_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    col4.download_button(
        "DOCX-отчет",
        data=hypotheses_to_docx(hypotheses, brief),
        file_name="hypotheses_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )
    col5.download_button(
        "PDF-отчет",
        data=hypotheses_to_pdf(hypotheses, brief),
        file_name="hypotheses_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _novelty_explanation(hypothesis) -> str:
    source_count = len({item.source for item in hypothesis.evidence})
    avg_relevance = sum(item.score for item in hypothesis.evidence) / max(len(hypothesis.evidence), 1)
    return (
        f"Новизна {hypothesis.novelty:.2f}: гипотеза сопоставлена с {source_count} источниками, "
        f"средняя похожесть evidence {avg_relevance:.3f}. Чем ниже прямое совпадение и выше механизм/эффект, "
        "тем выше приоритет для экспертной проверки."
    )


def _render_feedback_controls(rank: int, hypothesis) -> None:
    st.markdown("**Экспертная оценка**")
    if not _can_edit():
        st.caption("Текущая роль может просматривать feedback, но не сохранять новые оценки.")
        return
    col1, col2, col3 = st.columns(3)
    if col1.button("Подтверждена", key=f"feedback_ok_{rank}"):
        _save_feedback(hypothesis, "confirmed")
        st.success("Оценка сохранена: подтверждена")
    if col2.button("Отклонена", key=f"feedback_bad_{rank}"):
        _save_feedback(hypothesis, "rejected")
        st.warning("Оценка сохранена: отклонена")
    if col3.button("На проверку", key=f"feedback_test_{rank}"):
        _save_feedback(hypothesis, "needs_validation")
        st.info("Оценка сохранена: на проверку")


def _render_tracker_controls(rank: int, hypothesis) -> None:
    trackers = configured_trackers()
    st.markdown("**Интеграция с таск-трекером**")
    if not _can_edit():
        st.caption("Создание задач доступно ролям expert/admin.")
        return
    col1, col2 = st.columns(2)
    if col1.button("Создать Jira task", key=f"jira_task_{rank}", disabled=not trackers["jira"]):
        try:
            url = create_jira_issue(hypothesis)
            st.success(f"Jira задача создана: {url}")
        except Exception as exc:
            st.error(f"Не удалось создать Jira задачу: {exc}")
    if col2.button("Создать YouTrack task", key=f"youtrack_task_{rank}", disabled=not trackers["youtrack"]):
        try:
            url = create_youtrack_issue(hypothesis)
            st.success(f"YouTrack задача создана: {url}")
        except Exception as exc:
            st.error(f"Не удалось создать YouTrack задачу: {exc}")
    if not any(trackers.values()):
        st.caption("Для прямой отправки задач задайте Jira или YouTrack переменные в `.env`.")


def _save_feedback(hypothesis, status: str) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if FEEDBACK_PATH.exists():
        try:
            existing = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    existing.append(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "title": hypothesis.title,
            "statement": hypothesis.statement,
            "tags": hypothesis.tags,
            "scores": {
                "total": hypothesis.total_score,
                "novelty": hypothesis.novelty,
                "feasibility": hypothesis.feasibility,
                "expected_value": hypothesis.expected_value,
                "risk": hypothesis.risk,
                "confidence": hypothesis.confidence,
            },
            "sources": sorted({item.source for item in hypothesis.evidence}),
            "calculations": [asdict(item) for item in hypothesis.calculations],
        }
    )
    FEEDBACK_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
