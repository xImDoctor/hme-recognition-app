from datetime import datetime


def export_to_txt(latex: str, filename: str = None) -> str:
    """
    Создает содержимое .txt файла с LaTeX результатом.

    Args:
        latex: LaTeX строка
        filename: Имя файла (опционально)

    Returns:
        Содержимое файла как строка
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""Результат распознавания рукописного математического выражения
Дата: {timestamp}

LaTeX:
{latex}
"""
    
    return content


def create_download_button_data(latex: str) -> bytes:
    """
    Создает байтовые данные для кнопки скачивания в Streamlit.

    Args:
        latex: LaTeX строка

    Returns:
        Байтовые данные файла
    """

    content = export_to_txt(latex)
    return content.encode("utf-8")

