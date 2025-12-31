import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

from src.model_loader import load_model_and_processor, get_model_info
from src.preprocessing import preprocess_image
from src.inference import predict_latex
from src.metrics import compute_metrics
from src.export import create_download_button_data


def render_recognition_tab(selected_model_key: str):
    """
    Рендерит главную вкладку распознавания с подтабами Canvas и загрузки изображения.

    Args:
        selected_model_key: Ключ выбранной модели
    """

    st.header("Распознавание рукописных математических выражений")

    # Загрузка модели
    model_info = get_model_info(selected_model_key)
    if not model_info:
        st.error("Не удалось загрузить информацию о модели")
        return

    processor, model = load_model_and_processor(model_info["path"])

    # Создание подтабов
    subtab1, subtab2 = st.tabs(["Canvas (рисование)", "Загрузка изображения"])

    with subtab1:
        render_canvas_subtab(processor, model)

    with subtab2:
        render_upload_subtab(processor, model)



def render_canvas_subtab(processor, model):
    """
    Рендерит подтаб с Canvas для рисования.
    """
    st.subheader("Нарисуйте математическое выражение")

    # Настройки предобработки
    with st.expander("Настройки предобработки"):
        col1, col2 = st.columns(2)
        with col1:
            apply_inversion = st.checkbox(
                "Автоинверсия (темный фон)",
                value=False,
                key="canvas_inversion"
            )
        with col2:
            apply_binarization = st.checkbox(
                "Бинаризация",
                value=False,
                key="canvas_binarization"
            )


    # Canvas для рисования
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=3,
        stroke_color="black",
        background_color="white",
        height=200,
        width=700,
        drawing_mode="freedraw",
        key="canvas_main",
    )

    # Кнопка распознавания
    col1, col2 = st.columns([1, 4])
    with col1:
        recognize_btn = st.button("Распознать", type="primary", key="canvas_recognize")

    # Обработка результата
    if recognize_btn:
        if canvas_result.image_data is None or np.sum(canvas_result.image_data) == 0:
            st.warning("Canvas пустой. Нарисуйте формулу перед распознаванием.")
        else:
            with st.spinner("Распознавание..."):
                # Конвертация canvas в PIL Image
                img_data = canvas_result.image_data[:, :, 0:3].astype(np.uint8)
                image = Image.fromarray(img_data, mode="RGB")

                # Предобработка
                processed_image = preprocess_image(
                    image,
                    apply_inversion=apply_inversion,
                    apply_binarization=apply_binarization
                )

                # Инференс
                latex = predict_latex(processed_image, processor, model)

                # Сохранение в session state
                st.session_state.canvas_result = {
                    "latex": latex,
                    "image": processed_image
                }

    # Отображение результатов
    if "canvas_result" in st.session_state:
        display_recognition_results(
            st.session_state.canvas_result["latex"],
            st.session_state.canvas_result["image"],
            key_prefix="canvas"
        )


def render_upload_subtab(processor, model):
    """
    Рендерит подтаб с загрузкой изображения.
    """
    st.subheader("Загрузите изображение математического выражения")

    # Настройки предобработки
    with st.expander("Настройки предобработки"):
        col1, col2 = st.columns(2)
        with col1:
            apply_inversion = st.checkbox(
                "Автоинверсия (темный фон)",
                value=False,
                key="upload_inversion"
            )
        with col2:
            apply_binarization = st.checkbox(
                "Бинаризация",
                value=False,
                key="upload_binarization"
            )

    # File uploader
    uploaded_file = st.file_uploader(
        "Выберите изображение (PNG, JPEG)",
        type=["png", "jpg", "jpeg"],
        key="image_uploader"
    )

    if uploaded_file is not None:
        # Загрузка изображения
        image = Image.open(uploaded_file)

        # Превью оригинального изображения
        col1, col2 = st.columns([2, 3])
        with col1:
            st.image(image, caption="Исходное изображение", use_container_width=True)

        # Кнопка распознавания
        with col2:
            recognize_btn = st.button("Распознать", type="primary", key="upload_recognize")

            if recognize_btn:
                with st.spinner("Распознавание..."):
                    # Предобработка
                    processed_image = preprocess_image(
                        image,
                        apply_inversion=apply_inversion,
                        apply_binarization=apply_binarization
                    )

                    # Инференс
                    latex = predict_latex(processed_image, processor, model)

                    # Сохранение в session state
                    st.session_state.upload_result = {
                        "latex": latex,
                        "image": processed_image
                    }

        # Отображение результатов
        if "upload_result" in st.session_state:
            display_recognition_results(
                st.session_state.upload_result["latex"],
                st.session_state.upload_result["image"],
                key_prefix="upload"
            )


def display_recognition_results(latex: str, image: Image.Image, key_prefix: str):
    """
    Отображает результаты распознавания в 3 форматах (код, рендеринг и экспорт .txt) + метрики.

    Args:
        latex: Предсказанная LaTeX строка
        image: Предобработанное изображение
        key_prefix: Префикс для ключей Streamlit виджетов
    """
    st.markdown("---")
    st.success("Распознавание завершено!")

    # Текст LaTeX (строка)
    st.markdown("### LaTeX код:")
    st.code(latex, language="latex")

    # Рендеринг через st.latex()
    st.markdown("### Рендеринг:")
    try:
        st.latex(latex)
    except Exception as e:
        st.warning(f"Не удалось отрендерить LaTeX: {str(e)}")
        st.text(latex)

    # Экспорт в .txt
    st.markdown("### Экспорт:")
    download_data = create_download_button_data(latex)
    st.download_button(
        label="Скачать результат (.txt)",
        data=download_data,
        file_name=f"recognition_result_{key_prefix}.txt",
        mime="text/plain",
        key=f"{key_prefix}_download"
    )

    # Метрики (опционально)
    st.markdown("---")
    st.markdown("### Оценка качества (опционально)")
    ground_truth = st.text_input(
        "Введите правильный LaTeX (ground truth) для вычисления метрик:",
        key=f"{key_prefix}_gt"
    )

    if ground_truth:
        metrics = compute_metrics(latex, ground_truth)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CER (Character Error Rate)", f"{metrics['cer']:.4f}")
        with col2:
            st.metric("Edit Distance", metrics['edit_distance'])
        with col3:
            if metrics['exact_match']:
                st.success("Точное совпадение!")
            else:
                st.error("Не совпадает")
