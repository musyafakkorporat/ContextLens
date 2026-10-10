import os
import time

import pandas as pd
import torch

from pii_redaction import redact_pii
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

DATA_FILE = (
    "data/processed/dataset_preprocessed.csv"
)

OUTPUT_FILE = (
    "results/contextlens_sample_50.csv"
)

TOXICITY_MODEL = (
    "results/toxicity_indobertweet_1000/final_model"
)

POLARIZATION_MODEL = (
    "results/polarized_indobertweet_1000/final_model"
)

GEMINI_MODEL = "gemini-3.5-flash-lite"

TOXICITY_THRESHOLD = 0.06

POLARIZATION_THRESHOLD = 0.50

SAMPLE_SIZE = 50


# =========================
# ENVIRONMENT
# =========================

load_dotenv()

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY tidak ditemukan di file .env"
    )

client = genai.Client(
    api_key=api_key
)


# =========================
# GEMINI OUTPUT SCHEMA
# =========================

class ContextAnalysis(BaseModel):

    summary: str

    claim: str

    content_type: Literal[
        "fact",
        "opinion",
        "prediction",
        "advertisement",
        "mixed",
        "unclear"
    ]

    reasoning_pattern: str

    evidence_needed: str


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    DATA_FILE
)

df = df[
    df["text_clean"].notna()
]

df = df[
    df["text_clean"].str.strip() != ""
]

df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
).reset_index(drop=True)


print("=== DATA ===")

print(
    "Jumlah teks:",
    len(df)
)


# =========================
# LOAD TOXICITY MODEL
# =========================

print("\n=== LOAD TOXICITY MODEL ===")

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


# =========================
# LOAD POLARIZATION MODEL
# =========================

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
# PREDICT CLASSIFICATION
# =========================

def predict_probability(
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

    return probabilities.numpy()


# =========================
# PROCESS DATA
# =========================

results = []


for index, row in df.iterrows():

    text = row["text_clean"]

    # Teks asli digunakan oleh model lokal.
    # Sebelum dikirim ke Gemini, PII disamarkan.
    text_for_llm = redact_pii(text)

    print(
        f"\n[{index + 1}/{len(df)}] "
        "Processing..."
    )


    # ---------------------
    # TOXICITY
    # ---------------------

    toxicity_prob = predict_probability(
        text,
        toxicity_tokenizer,
        toxicity_model
    )

    toxicity_score = float(
        toxicity_prob[1]
    )

    toxicity_label = int(
        toxicity_score >= TOXICITY_THRESHOLD
    )


    # ---------------------
    # POLARIZATION
    # ---------------------

    polarization_prob = predict_probability(
        text,
        polarization_tokenizer,
        polarization_model
    )

    polarization_score = float(
        polarization_prob[1]
    )

    polarization_label = int(
        polarization_score >= POLARIZATION_THRESHOLD
    )


    # ---------------------
    # GEMINI
    # ---------------------

    prompt = f"""
Analisis teks berikut secara netral.

TEKS:
{text_for_llm}

HASIL MODEL NLP:

Toxicity:
- label: {toxicity_label}
- probability: {toxicity_score:.4f}

Polarization:
- label: {polarization_label}
- probability: {polarization_score:.4f}

Gunakan hasil model hanya sebagai sinyal tambahan,
bukan sebagai kebenaran mutlak.

Tugas:
1. Buat ringkasan singkat.
2. Identifikasi klaim utama jika ada.
3. Tentukan jenis isi:
   - fact
   - opinion
   - prediction
   - advertisement
   - mixed
   - unclear
4. Jelaskan kemungkinan pola penalaran yang terlihat.
   Jangan menyatakan bahwa pola tersebut pasti merupakan
   logical fallacy.
5. Jelaskan bukti atau konteks yang diperlukan untuk
   memeriksa klaim tersebut.

Untuk iklan atau promosi, gunakan "advertisement".
Jika konteks terlalu sedikit untuk menentukan jenis isi,
gunakan "unclear".

Jangan mengarang fakta, sumber, atau bukti.
"""

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": (
                    "Kamu adalah analis informasi yang netral. "
                    "Jangan memihak pihak atau kelompok tertentu. "
                    "Bedakan fakta, opini, dan klaim yang belum "
                    "terverifikasi. "
                    "Reasoning pattern hanya merupakan kemungkinan, "
                    "bukan diagnosis atau kepastian."
                ),
                "response_mime_type": "application/json",
                "response_schema": ContextAnalysis
            }
        )

        analysis = ContextAnalysis.model_validate_json(
            response.text
        )

    except Exception as e:

        print(
            "Gemini gagal:",
            str(e)
        )

        analysis = ContextAnalysis(
            summary="Analisis Gemini gagal diproses.",
            claim="",
            content_type="unclear",
            reasoning_pattern="",
            evidence_needed=""
        )


    # ---------------------
    # SAVE RESULT
    # ---------------------

    results.append({

        "text_id": row["text_id"],

        "text": row["text"],

        "text_redacted": text_for_llm,

        "toxicity": toxicity_label,

        "toxicity_probability": round(
            toxicity_score,
            4
        ),

        "polarization": polarization_label,

        "polarization_probability": round(
            polarization_score,
            4
        ),

        "summary": analysis.summary,

        "claim": analysis.claim,

        "content_type": analysis.content_type,

        "reasoning_pattern": (
            analysis.reasoning_pattern
        ),

        "evidence_needed": (
            analysis.evidence_needed
        )
    })


    # Jeda kecil agar request Gemini
    # tidak dikirim terlalu cepat.
    time.sleep(2)


# =========================
# SAVE CSV
# =========================

result_df = pd.DataFrame(
    results
)

os.makedirs(
    "results",
    exist_ok=True
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================
# FINAL OUTPUT
# =========================

print(
    "\n=============================="
)

print(
    "BATCH SELESAI"
)

print(
    "=============================="
)

print(
    "File:",
    OUTPUT_FILE
)

print(
    "Jumlah:",
    len(result_df)
)