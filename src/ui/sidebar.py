import streamlit as st
from src.model_loader import load_models_config, get_model_info


def render_sidebar() -> str:
    """
    Рендерит сайдбар с выбором модели и описанием.

    Returns:
        Ключ выбранной модели
    """
    st.sidebar.title("Настройки")

    # Загрузка доступных моделей
    models_config = load_models_config()
    if not models_config:
        st.sidebar.error("Не удалось загрузить конфигурацию моделей")
        return None

    model_names = {key: info["name"] for key, info in models_config.items()}

    # выбор модели из выпадающего селектбокса
    selected_model_key = st.sidebar.selectbox(
        "Выберите модель:",
        options=list(model_names.keys()),
        format_func=lambda x: model_names[x]
    )

    # Информация о модели
    model_info = get_model_info(selected_model_key)
    if model_info and "metrics" in model_info:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Метрики модели")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("ExpRate≤2", f"{model_info['metrics']['exp_rate_2']:.2f}%")
        with col2:
            st.metric("CER", f"{model_info['metrics']['cer']:.2f}%")

        st.sidebar.metric("Avg Edit Distance", f"{model_info['metrics']['avg_edit_distance']:.2f}")

    # Описание приложения
    st.sidebar.markdown("---")
    st.sidebar.markdown("### О приложении")
    st.sidebar.info(
        "Приложение для распознавания рукописных математических выражений "
        "с использованием трансорфмерной модели TrOCR, дообученной на датасете HME100K."
    )

    return selected_model_key
