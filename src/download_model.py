"""
Модуль для загрузки модели с Google Drive при первом запуске.
"""
import os
import streamlit as st
import gdown
from pathlib import Path


def download_model_from_gdrive():
    """
    Загружает модель с Google Drive, если её нет локально.
    Использует st.secrets для получения ссылки на модель.
    """
    model_path = Path("models/trocr1-5ep")

    # Существует ли модель локально
    if model_path.exists() and (model_path / "config.json").exists():
        return

    # Получаем ссылку из secrets
    try:
        gdrive_folder_id = st.secrets["model"]["gdrive_folder_id"]
    except KeyError:
        st.error("Ссылка на модель не найдена!")
        st.stop()

    # Создаем директорию для модели
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Загружаем модель с показом прогресса
    with st.spinner("Загрузка модели из внешнего хранилища... Пожалуйста подождите"):
        try:
            # URL для gdown
            gdrive_url = f"https://drive.google.com/drive/folders/{gdrive_folder_id}"

            # Загружаем папку
            gdown.download_folder(
                url=gdrive_url,
                output=str(model_path),
                quiet=False,
                use_cookies=False
            )

            st.success("Модель успешно загружена!")
        except Exception as e:
            st.error(f"Ошибка при загрузке модели: {str(e)}")
            st.stop()
