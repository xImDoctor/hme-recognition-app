
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Вычисляет расстояние Левенштейна между двумя строками.
    Даёт edit distance метрику для вывода в приложении.

    Args:
        s1: Первая строка
        s2: Вторая строка

    Returns:
        Расстояние Левенштейна (количество операций редактирования: удалений, вставок и замен)
    """
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[len1][len2]


def compute_cer(prediction: str, reference: str) -> float:
    """
    Вычисляет Character Error Rate (CER) для вывода в приложении.
    Представляет собой посимвольную ошибку при предсказании формулы.

    Args:
        prediction: Предсказанная строка
        reference: Референсная строка (ground truth)

    Returns:
        CER (от 0)
    """
    if len(reference) == 0:
        return 0.0 if len(prediction) == 0 else 1.0

    distance = levenshtein_distance(prediction, reference)
    return distance / len(reference)



def compute_metrics(prediction: str, ground_truth: str) -> dict:
    """
    Вычисляет все метрики для пары предсказание-ground truth:
    CER, Edit Distance и полное совпадение (есть или нет, булево значение).

    Args:
        prediction: Предсказанная LaTeX строка
        ground_truth: Ground truth LaTeX строка

    Returns:
        dict с метриками: {"cer": float, "edit_distance": int, "exact_match": bool}
    """
    cer = compute_cer(prediction, ground_truth)
    edit_distance = levenshtein_distance(prediction, ground_truth)
    exact_match = (prediction == ground_truth)

    return {
        "cer": cer,
        "edit_distance": edit_distance,
        "exact_match": exact_match
    }
