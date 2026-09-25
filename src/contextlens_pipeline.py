import os

import pandas as pd
import torch

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from google import genai


# =========================
# CONFIG
# =========================

TOXICITY_MODEL = (
    "results/toxicity_indobertweet_1000/final_model"
)

POLARIZATION_MODEL = (
    "results/polarized_indobertweet_1000/final_model"
)

GEMINI_MODEL = "gemini-3.5-flash-lite"

TOXICITY_THRESHOLD = 0.06


# =========================
# LOAD ENV
# =========================

load_dotenv()

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY belum ditemukan di .env"
    )

client = genai.Client(
    api_key=api_key
)


# =========================
# GEMINI OUTPUT
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
# LOAD INDoBERTWEET
# =========================

print("=== LOAD TOXICITY MODEL ===")

toxicity_tokenizer = (
    AutoTokenizer.from_pretrained(
        TOXICITY_MODEL
    )
)

toxicity_model = (
    AutoModelForSequenceClassification.from_pretrained(
        TOXICITY_MODEL
    )
)

toxicity_model.eval()

print("Toxicity model siap.")


print("\n=== LOAD POLARIZATION MODEL ===")

polarization_tokenizer = (
    AutoTokenizer.from_pretrained(
        POLARIZATION_MODEL
    )
)

polarization_model = (
    AutoModelForSequenceClassification.from_pretrained(
        POLARIZATION_MODEL
    )
)

polarization_model.eval()

print("Polarization model siap.")


# =========================
# TEXT INPUT
# =========================

text = """
Banyak orang bilang belajar programming sudah tidak penting
karena AI sekarang bisa membuat kode sendiri. Menurut saya,
mahasiswa tetap perlu belajar programming supaya memahami
bagaimana software bekerja.
"""


# =========================
# HELPER
# =========================

def predict_class(
    text,
    tokenizer,
    model
):

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

    probabilities = torch.softmax(
        output.logits,
        dim=1
    )[0]

    prediction = (
        torch.argmax(probabilities)
        .item()
    )

    return (
        prediction,
        probabilities.numpy()
    )


# =========================
# TOXICITY
# =========================

print("\n=== TOXICITY ===")

toxicity_prediction, toxicity_prob = (
    predict_class(
        text,
        toxicity_tokenizer,
        toxicity_model
    )
)

toxicity_score = float(
    toxicity_prob[1]
)

toxicity_result = int(
    toxicity_score >= TOXICITY_THRESHOLD
)


# =========================
# POLARIZATION
# =========================

print("\n=== POLARIZATION ===")

polarization_prediction, polarization_prob = (
    predict_class(
        text,
        polarization_tokenizer,
        polarization_model
    )
)


# =========================
# DISPLAY ML RESULTS
# =========================

print(
    "Toxicity probability:",
    round(toxicity_score, 4)
)

print(
    "Toxicity:",
    toxicity_result
)

print(
    "Polarization probability:",
    round(
        float(polarization_prob[1]),
        4
    )
)

print(
    "Polarization:",
    polarization_prediction
)


# =========================
# GEMINI
# =========================

print("\n=== GEMINI ANALYSIS ===")

prompt = f"""
Analisis teks berikut secara netral.

TEKS:
{text}

Hasil model NLP sebelumnya:

Toxicity:
- label: {toxicity_result}
- probability: {toxicity_score:.4f}

Polarization:
- label: {polarization_prediction}
- probability: {float(polarization_prob[1]):.4f}

Gunakan hasil model tersebut hanya sebagai sinyal tambahan,
bukan sebagai kebenaran mutlak.

Tugas:
1. Buat ringkasan singkat.
2. Identifikasi klaim utama jika ada.
3. Tentukan apakah isi utama berupa fact, opinion, prediction,
   atau mixed.
4. Jelaskan kemungkinan pola penalaran yang terlihat.
   Jangan menyatakan bahwa pola tersebut pasti merupakan
   logical fallacy.
5. Jelaskan bukti atau konteks apa yang dibutuhkan untuk
   memeriksa klaim tersebut.

Jangan mengarang sumber, fakta, atau bukti yang tidak ada
dalam teks.
"""


response = client.models.generate_content(

    model=GEMINI_MODEL,

    contents=prompt,

    config={
        "system_instruction": (
            "Kamu adalah analis informasi yang netral. "
            "Jangan memihak tokoh, kelompok, negara, partai, "
            "atau pandangan politik tertentu. "
            "Bedakan fakta dari opini dan klaim yang belum "
            "terverifikasi. "
            "Gunakan bahasa yang hati-hati."
        ),

        "response_mime_type": "application/json",

        "response_schema": ContextAnalysis
    }
)


result = ContextAnalysis.model_validate_json(
    response.text
)


# =========================
# FINAL RESULT
# =========================

print("\n==============================")
print("CONTEXTLENS RESULT")
print("==============================")

print(
    "\n[Toxicity]"
)

print(
    "Label:",
    toxicity_result
)

print(
    "Probability:",
    round(
        toxicity_score,
        4
    )
)


print(
    "\n[Polarization]"
)

print(
    "Label:",
    polarization_prediction
)

print(
    "Probability:",
    round(
        float(polarization_prob[1]),
        4
    )
)


print(
    "\n[Gemini]"
)

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