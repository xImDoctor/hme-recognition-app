import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


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
