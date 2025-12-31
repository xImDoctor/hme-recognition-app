import cv2
import numpy as np
from PIL import Image


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Конвертирует изображение в RGB (базовая предобработка).

    Args:
        image: PIL Image

    Returns:
        PIL Image в формате RGB
    """
    return image.convert("RGB")


def auto_invert(image: Image.Image) -> Image.Image:
    """
    Автоматическая инверсия для изображений с темным фоном.
    Опциональна в основном приложении.

    Args:
        image: PIL Image

    Returns:
        Инвертированное изображение (если средняя яркость < 128)
    """
    gray = np.array(image.convert("L"))
    mean_brightness = np.mean(gray)

    if mean_brightness < 128:
        # Темный фон - инвертируем
        inverted = 255 - gray
        return Image.fromarray(inverted).convert("RGB")

    return image


def binarize(image: Image.Image, threshold: int = 0) -> Image.Image:
    """
    Бинаризация изображения (Otsu или по фиксированному порогу).
    Опциональна в основном приложении.

    Args:
        image: PIL Image
        threshold: Порог бинаризации (если 0, использует Otsu)

    Returns:
        Бинаризованное изображение
    """
    gray = np.array(image.convert("L"))

    if threshold == 0:
        # Otsu's method
        threshold_value, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return Image.fromarray(binary).convert("RGB")


def preprocess_image(image: Image.Image, apply_inversion: bool = False, apply_binarization: bool = False, 
                     binarization_threshold: int = 0) -> Image.Image:
    """
    Применяет выбранные шаги предобработки к изображению.

    Args:
        image: Исходное изображение
        apply_inversion: Применить автоинверсию
        apply_binarization: Применить бинаризацию
        binarization_threshold: Порог бинаризации (0 для Otsu)

    Returns:
        Предобработанное изображение
    """
    # Базовая конвертация в RGB
    processed = convert_to_rgb(image)

    # Опциональная инверсия
    if apply_inversion:
        processed = auto_invert(processed)

    # Опциональная бинаризация
    if apply_binarization:
        processed = binarize(processed, binarization_threshold)

    return processed


