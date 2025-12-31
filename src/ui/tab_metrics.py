import streamlit as st
import json
from pathlib import Path
import pandas as pd


def render_metrics_tab():
    """
    Рендерит вкладку "Метрики обучения" (исключаемый модуль).
    """
    st.header("Метрики обучения модели")

    # Путь к local-docs
    docs_path = Path(__file__).parent.parent.parent / "local-docs"

    # Проверка наличия данных
    history_file = docs_path / "training_history.json"
    if not history_file.exists():
        st.warning("Файл training_history.json не найден. Метрики обучения недоступны.")
        st.info(f"Ожидаемый путь: {history_file}")
        return

    # Загрузка истории обучения
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception as e:
        st.error(f"Ошибка загрузки training_history.json: {e}")
        return

    # Раздел 1: Финальные метрики
    st.markdown("## Финальные результаты (эпоха 5)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expression Rate", f"{history['val_exp_rate'][-1]:.2f}%")
    with col2:
        st.metric("CER", f"{history['val_cer'][-1]*100:.2f}%")
    with col3:
        st.metric("Avg Edit Distance", f"{history['val_avg_edit_distance'][-1]:.2f}")
    with col4:
        st.metric("Validation Loss", f"{history['val_loss'][-1]:.4f}")

    # Раздел 2: Таблица с историей по эпохам
    st.markdown("---")
    st.markdown("## История обучения по эпохам")

    df = pd.DataFrame({
        "Эпоха": list(range(1, len(history['train_loss']) + 1)),
        "Train Loss": history['train_loss'],
        "Val Loss": history['val_loss'],
        "ExpRate (%)": history['val_exp_rate'],
        "CER (%)": [x * 100 for x in history['val_cer']],
        "Edit Distance": history['val_avg_edit_distance'],
        "Learning Rate": history['learning_rate']
    })

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Раздел 3: Графики
    st.markdown("---")
    st.markdown("## Графики обучения")

    # Проверка наличия графиков
    graphs = {
        "Comprehensive Metrics": "comprehensive_metrics.png",
        "Quality Metrics": "quality_metrics.png",
        "Loss Dynamics": "loss_dynamics.png",
        "Expression Recognition Rate": "expression_recognition_rate.png",
        "Epoch Improvements": "epoch_improvements.png",
        "Learning Rate Schedule": "learning_rate_schedule.png",
        "Before/After Comparison": "before_after_comparison.png"
    }

    # Отображение графиков в 2 колонки
    graph_files = [
        (name, docs_path / filename)
        for name, filename in graphs.items()
        if (docs_path / filename).exists()
    ]

    if not graph_files:
        st.info("Графики обучения не найдены в папке local-docs/")
    else:
        for i in range(0, len(graph_files), 2):
            col1, col2 = st.columns(2)

            with col1:
                if i < len(graph_files):
                    name, filepath = graph_files[i]
                    st.markdown(f"### {name}")
                    st.image(str(filepath), use_container_width=True)

            with col2:
                if i + 1 < len(graph_files):
                    name, filepath = graph_files[i + 1]
                    st.markdown(f"### {name}")
                    st.image(str(filepath), use_container_width=True)

    # Раздел 4: Анализ улучшений
    st.markdown("---")
    st.markdown("## Анализ улучшений")

    # Вычисление улучшений от эпохи 1 до эпохи 5
    exp_rate_improvement = history['val_exp_rate'][-1] - history['val_exp_rate'][0]
    cer_improvement = history['val_cer'][0] - history['val_cer'][-1]
    edit_dist_improvement = history['val_avg_edit_distance'][0] - history['val_avg_edit_distance'][-1]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "ExpRate улучшение",
            f"+{exp_rate_improvement:.2f}%",
            delta=f"{exp_rate_improvement:.2f}%"
        )
    with col2:
        st.metric(
            "CER снижение",
            f"-{cer_improvement*100:.2f}%",
            delta=f"{-cer_improvement*100:.2f}%",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            "Edit Distance снижение",
            f"-{edit_dist_improvement:.2f}",
            delta=f"{-edit_dist_improvement:.2f}",
            delta_color="inverse"
        )
