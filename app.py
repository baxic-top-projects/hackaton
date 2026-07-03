from __future__ import annotations

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

from src.exporters import hypotheses_to_frame, hypotheses_to_json, hypotheses_to_markdown
from src.graphing import build_relation_graph, graph_to_plotly_data
from src.hypothesis_engine import generate_hypotheses
from src.ingestion import chunk_documents, load_documents_from_paths, load_uploaded_files
from src.models import ResearchBrief
from src.retrieval import KnowledgeIndex
from src.yandex_ai import generate_expert_summary, is_yandex_configured


SAMPLE_DIR = Path("data/sample_knowledge")


load_dotenv()


def main() -> None:
    st.set_page_config(page_title="Фабрика гипотез", page_icon="H", layout="wide")
    st.title("Фабрика гипотез")
    st.caption("Интерпретируемый RAG-прототип для генерации и приоритизации НИОКР-гипотез")

    with st.sidebar:
        st.header("Входные данные")
        target = st.text_area(
            "Целевое свойство или технологическая проблема",
            value="Повысить жаропрочность никелевого сплава на 15% без существенного снижения пластичности",
            height=90,
        )
        constraints = st.text_area(
            "Ограничения",
            value="Использовать доступные легирующие элементы; избегать сильного роста себестоимости; лабораторная проверка за 2 недели",
            height=90,
        )
        available_materials = st.text_input(
            "Доступное сырье",
            value="никель, хром, ниобий, титан, молибден, бор",
        )
        equipment = st.text_input(
            "Оборудование",
            value="вакуумная плавка, печь отжига, установка старения, испытательная машина",
        )
        budget = st.text_input("Бюджет/сроки", value="средний бюджет, 2 недели на первичный скрининг")

        st.subheader("Веса ранжирования")
        novelty = st.slider("Новизна", 0.0, 1.0, 0.20, 0.05)
        feasibility = st.slider("Реализуемость", 0.0, 1.0, 0.25, 0.05)
        expected_value = st.slider("Ожидаемая ценность", 0.0, 1.0, 0.30, 0.05)
        risk = st.slider("Низкий риск", 0.0, 1.0, 0.15, 0.05)
        confidence = st.slider("Уверенность источников", 0.0, 1.0, 0.10, 0.05)

        uploaded_files = st.file_uploader(
            "Загрузить базу знаний",
            type=["txt", "md", "pdf", "docx", "csv", "xlsx"],
            accept_multiple_files=True,
        )
        limit = st.slider("Количество гипотез", 3, 12, 6)
        use_yandex = st.checkbox(
            "Добавить экспертную сводку YandexGPT",
            value=is_yandex_configured(),
            disabled=not is_yandex_configured(),
        )
        if not is_yandex_configured():
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

    if not run:
        st.info("Нажмите кнопку генерации. Если файлы не загружены, будет использована встроенная демо-база знаний.")
        _render_requirements_match()
        return

    with st.spinner("Индексирую источники и строю гипотезы..."):
        documents = load_uploaded_files(uploaded_files) if uploaded_files else load_documents_from_paths(SAMPLE_DIR.glob("*"))
        chunks = chunk_documents(documents)
        if not chunks:
            st.error("Не удалось извлечь текст из источников. Загрузите файлы или добавьте демо-данные.")
            return
        index = KnowledgeIndex.build(chunks)
        hypotheses = generate_hypotheses(brief, index, limit=limit)

    st.success(f"Сформировано гипотез: {len(hypotheses)}. Источников: {len(documents)}. Фрагментов: {len(chunks)}.")

    if use_yandex:
        _render_yandex_summary(brief, hypotheses)
    _render_rank_table(hypotheses)
    _render_graph(hypotheses)
    _render_cards(hypotheses)
    _render_exports(hypotheses, brief)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def _render_requirements_match() -> None:
    st.subheader("Что закрывает прототип")
    col1, col2, col3 = st.columns(3)
    col1.metric("Входы", "PDF/DOCX/XLSX/TXT")
    col2.metric("Объяснимость", "цитаты + скоринг")
    col3.metric("Экспорт", "JSON/CSV/MD")
    st.write(
        "Система принимает цель, ограничения и базу знаний, извлекает релевантные фрагменты, "
        "генерирует проверяемые гипотезы, ранжирует их по настраиваемым критериям и формирует план проверки."
    )


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
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


def _render_yandex_summary(brief: ResearchBrief, hypotheses) -> None:
    st.subheader("Экспертная сводка YandexGPT")
    try:
        with st.spinner("Запрашиваю экспертную сводку в Yandex AI Studio..."):
            summary = generate_expert_summary(brief, hypotheses)
        st.info(summary)
    except Exception as exc:
        st.warning(f"YandexGPT недоступен, базовая генерация сохранена: {exc}")


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

            st.markdown("**Источники**")
            for item in hypothesis.evidence:
                st.caption(f"{item.source} · релевантность {item.score:.3f}")
                st.write(item.quote)


def _render_exports(hypotheses, brief: ResearchBrief) -> None:
    st.subheader("Экспорт")
    frame = hypotheses_to_frame(hypotheses)
    col1, col2, col3 = st.columns(3)
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


if __name__ == "__main__":
    main()
