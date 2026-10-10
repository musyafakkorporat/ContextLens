import re


def redact_pii(text):
    """
    Menyamarkan informasi pribadi sederhana
    sebelum teks dikirim ke LLM.
    """

    text = str(text)

    # Email
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text
    )

    # URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "[URL]",
        text
    )

    # Nomor telepon Indonesia sederhana
    text = re.sub(
        r"(?<!\d)(?:\+62|62|0)8[\d\s-]{7,14}(?!\d)",
        "[PHONE]",
        text
    )

    return text