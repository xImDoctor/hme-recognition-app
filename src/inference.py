import torch
import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def predict_latex_unified(
    image: Image.Image,
    processor: TrOCRProcessor = None,
    model: VisionEncoderDecoderModel = None,
    max_length: int = 256,
    num_beams: int = 4
) -> str:
    """
    Универсальная функция распознавания.
    Автоматически выбирает между HF API и локальным инференсом.

    Приоритеты:
    1. HuggingFace API (если настроен в secrets)
    2. Локальная модель (если загружена)

    Args:
        image: PIL Image
        processor: TrOCRProcessor (опционально, для локального режима)
        model: VisionEncoderDecoderModel (опционально, для локального режима)
        max_length: Максимальная длина генерации
        num_beams: Количество лучей для beam search

    Returns:
        str: Распознанная LaTeX строка
    """
    # Проверяем наличие HF API в secrets
    try:
        use_hf_api = "huggingface" in st.secrets and "model_name" in st.secrets["huggingface"]
    except:
        use_hf_api = False

    if use_hf_api:
        # Режим 1: Используем HuggingFace API
        from src.inference_hf import predict_latex_hf
        return predict_latex_hf(image)
    else:
        # Режим 2: Используем локальный инференс
        if processor is None or model is None:
            st.error("Модель не загружена для локального инференса!")
            st.stop()
        return predict_latex(image, processor, model, max_length, num_beams)


def predict_latex(
    image: Image.Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    max_length: int = 256,
    num_beams: int = 4
) -> str:
    """
    Выполняет инференс на изображении и возвращает LaTeX строку.

    Args:
        image: PIL Image в формате RGB
        processor: TrOCRProcessor
        model: VisionEncoderDecoderModel
        max_length: Максимальная длина последовательности
        num_beams: Количество beams для beam search

    Returns:
        str: Предсказанная LaTeX строка
    """
    # Предобработка
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Генерация
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

    # Декодирование
    latex = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return latex
