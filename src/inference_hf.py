"""
Модуль инференса через HuggingFace Inference API.
Используется для экономии памяти - модель запускается на серверах HF.
"""
import streamlit as st
import requests
from PIL import Image
import io
import base64


def predict_latex_hf(image: Image.Image, model_name: str = None) -> str:
    """
    Выполняет распознавание математического выражения через HuggingFace Inference API.

    Args:
        image: PIL Image объект
        model_name: Имя модели на HuggingFace Hub (например, "your-username/trocr-hme-finetuned")
                   Если None, берется из st.secrets

    Returns:
        str: Распознанная LaTeX строка
    """
    # Получаем конфигурацию из secrets
    try:
        if model_name is None:
            model_name = st.secrets["huggingface"]["model_name"]
        hf_token = st.secrets["huggingface"].get("api_token", None)
    except KeyError:
        st.error("HuggingFace конфигурация не найдена в secrets!")
        st.stop()

    # Конвертируем изображение в base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # API endpoint
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"

    # Headers
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    # Отправляем запрос
    try:
        response = requests.post(
            api_url,
            headers=headers,
            data=img_bytes,
            timeout=30
        )
        response.raise_for_status()

        # Парсим ответ
        result = response.json()

        # HF Inference API возвращает разные форматы в зависимости от модели
        # Для image-to-text обычно: [{"generated_text": "..."}]
        if isinstance(result, list) and len(result) > 0:
            latex = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            latex = result.get("generated_text", result.get("text", ""))
        else:
            latex = str(result)

        return latex.strip()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            st.error("Модель загружается на HuggingFace. Попробуйте через 20 секунд.")
        elif e.response.status_code == 401:
            st.error("Неверный HuggingFace API токен.")
        else:
            st.error(f"Ошибка HuggingFace API: {e.response.status_code} - {e.response.text}")
        st.stop()

    except requests.exceptions.Timeout:
        st.error("Превышено время ожидания ответа от HuggingFace.")
        st.stop()

    except Exception as e:
        st.error(f"Ошибка при обращении к HuggingFace API: {str(e)}")
        st.stop()


def check_hf_model_status(model_name: str, hf_token: str = None) -> dict:
    """
    Проверяет статус модели на HuggingFace Hub.

    Returns:
        dict: Информация о статусе модели
    """
    api_url = f"https://api-inference.huggingface.co/status/{model_name}"

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}
