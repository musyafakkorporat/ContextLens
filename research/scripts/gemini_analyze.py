import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai


# =========================
# LOAD API KEY
# =========================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY belum ditemukan di file .env"
    )


client = genai.Client(
    api_key=api_key
)


# =========================
# OUTPUT STRUCTURE
# =========================

class ContextAnalysis(BaseModel):

    summary: str

    claim: str

    content_type: Literal[
        "fact",
        "opinion",
        "prediction",
        "mixed"
    ]

    reasoning_pattern: str

    evidence_needed: str


# =========================
# INPUT TEXT
# =========================

text = """
Banyak orang mengatakan bahwa belajar programming
sudah tidak penting karena sekarang AI bisa membuat kode.
Menurut saya, mahasiswa tetap perlu belajar programming
agar memahami bagaimana software bekerja.
"""


# =========================
# GEMINI REQUEST
# =========================

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=text,
    config={
        "system_instruction": (
            "Kamu adalah analis informasi yang netral. "
            "Analisis hanya berdasarkan teks yang diberikan. "
            "Bedakan fakta, opini, prediksi, dan campuran. "
            "Reasoning pattern hanya disebut sebagai kemungkinan, "
            "bukan diagnosis pasti. "
            "Jangan mengarang bukti yang tidak ada."
        ),
        "response_mime_type": "application/json",
        "response_schema": ContextAnalysis,
    }
)


# =========================
# RESULT
# =========================

result = ContextAnalysis.model_validate_json(
    response.text
)


print("\n=== HASIL ANALISIS ===")

print(
    "Summary:",
    result.summary
)

print(
    "Claim:",
    result.claim
)

print(
    "Content type:",
    result.content_type
)

print(
    "Reasoning pattern:",
    result.reasoning_pattern
)

print(
    "Evidence needed:",
    result.evidence_needed
)