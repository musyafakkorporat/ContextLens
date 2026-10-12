from collections import Counter


def is_repetitive_text(text: str) -> bool:
    """Return True for extremely repetitive text."""
    normalized_chars = "".join(
        char.casefold()
        for char in text
        if char.isalnum()
    )

    if len(normalized_chars) >= 20:
        counts = Counter(normalized_chars)
        dominant_ratio = counts.most_common(1)[0][1] / len(normalized_chars)

        if dominant_ratio >= 0.80:
            return True

    tokens = [
        token.casefold()
        for token in text.split()
        if any(char.isalnum() for char in token)
    ]

    if len(tokens) >= 10:
        counts = Counter(tokens)
        dominant_ratio = counts.most_common(1)[0][1] / len(tokens)

        if dominant_ratio >= 0.80:
            return True

    return False
