import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import json
from pathlib import Path


@st.cache_resource
def load_model_and_processor(model_path: str):
    """
    Загружает модель TrOCR и процессор с кешированием.

    Args:
        model_path: Путь к папке с моделью

    Returns:
        tuple: (processor, model)
    """
    try:
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        return processor, model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        st.info(f"Проверьте, что модель находится в папке: {model_path}")
        raise


def load_models_config() -> dict:
    """
    Загружает конфигурацию доступных моделей.

    Returns:
        dict: Конфигурация моделей
    """
    config_path = Path(__file__).parent.parent / "config" / "models.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Файл конфигурации моделей не найден: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"Ошибка парсинга JSON в {config_path}: {e}")
        return {}


def get_model_info(model_key: str) -> dict:
    """
    Получает информацию о конкретной модели.

    Args:
        model_key: Ключ модели в конфигурации

    Returns:
        dict: Информация о модели (путь, описание, метрики)
    """
    config = load_models_config()
    if model_key not in config:
        st.warning(f"Модель '{model_key}' не найдена в конфигурации")
        return {}
    return config.get(model_key, {})
