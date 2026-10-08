import os

import numpy as np
import pandas as pd
import torch

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sentence_transformers import SentenceTransformer

from google import genai

from calibration_config import (
    TOXICITY_TEMPERATURE,
    POLARIZATION_TEMPERATURE,
    TOXICITY_RAW_THRESHOLD,
    POLARIZATION_RAW_THRESHOLD,
    TOXICITY_UNCERTAIN_DELTA,
    POLARIZATION_UNCERTAIN_DELTA
)


# =========================
# CONFIG
# =========================

TOXICITY_MODEL = (
    "results/toxicity_indobertweet_5000/final_model"
)

POLARIZATION_MODEL = (
    "results/polarized_indobertweet_5000/final_model"
)

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

EMBEDDINGS_FILE = (
    "data/reference/mafindo_embeddings.npy"
)

METADATA_FILE = (
    "data/reference/mafindo_metadata.csv"
)

GEMINI_MODEL = "gemini-3.5-flash-lite"

TOXICITY_THRESHOLD = TOXICITY_RAW_THRESHOLD

MAFINDO_MIN_SCORE = 0.62

MAFINDO_TOP_K = 5

MAFINDO_MAX_CONTEXT = 3


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
# LOAD INDOBERTWEET
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
# LOAD MAFINDO
# =========================

print("\n=== LOAD MAFINDO REFERENCE ===")

mafindo_embeddings = np.load(
    EMBEDDINGS_FILE
)

mafindo_metadata = pd.read_csv(
    METADATA_FILE
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "MAfindo reference:",
    len(mafindo_metadata),
    "data"
)

print(
    "Embedding shape:",
    mafindo_embeddings.shape
)


# =========================
# TEXT INPUT
# =========================

def validate_text(text):

    text = str(
        text
    ).strip()

    if not text:
        raise ValueError(
            "Teks tidak boleh kosong."
        )

    return text


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


def calibrate_probability(
    probability,
    temperature
):

    probability = np.clip(
        float(probability),
        1e-7,
        1 - 1e-7
    )

    logit = np.log(
        probability / (1 - probability)
    )

    calibrated_logit = (
        logit / temperature
    )

    calibrated_probability = (
        1 / (1 + np.exp(-calibrated_logit))
    )

    return float(
        calibrated_probability
    )


# =========================
# MAFINDO SEARCH
# =========================

def search_mafindo(text):

    query_embedding = embedding_model.encode(
        [text],
        normalize_embeddings=True
    )[0]

    scores = (
        mafindo_embeddings
        @ query_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:MAFINDO_TOP_K]

    results = []

    for index in top_indices:

        score = float(
            scores[index]
        )

        if score < MAFINDO_MIN_SCORE:
            continue

        row = mafindo_metadata.iloc[index]

        results.append({
            "title": str(row["title"]),
            "content": str(row["content"]),
            "is_hoax": int(row["is_hoax"]),
            "score": score
        })

        if len(results) >= MAFINDO_MAX_CONTEXT:
            break

    return results


# =========================
# ANALYZE TEXT
# =========================

def analyze_text(text):

    text = validate_text(
        text
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

    # -------------------------
    # TOXICITY CALIBRATION
    # -------------------------

    toxicity_calibrated_score = calibrate_probability(
        toxicity_score,
        TOXICITY_TEMPERATURE
    )

    toxicity_calibrated_threshold = calibrate_probability(
        TOXICITY_RAW_THRESHOLD,
        TOXICITY_TEMPERATURE
    )

    # -------------------------
    # TOXICITY UNCERTAINTY
    # -------------------------

    toxicity_uncertain = (
        abs(
            toxicity_calibrated_score
            - toxicity_calibrated_threshold
        )
        < TOXICITY_UNCERTAIN_DELTA
    )

    toxicity_status = (
        "uncertain"
        if toxicity_uncertain
        else "confident"
    )

    # -------------------------
    # TOXICITY LABEL
    # -------------------------

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

    polarization_score = float(
        polarization_prob[1]
    )

    # -------------------------
    # POLARIZATION CALIBRATION
    # -------------------------

    polarization_calibrated_score = calibrate_probability(
        polarization_score,
        POLARIZATION_TEMPERATURE
    )

    polarization_calibrated_threshold = calibrate_probability(
        POLARIZATION_RAW_THRESHOLD,
        POLARIZATION_TEMPERATURE
    )

    # -------------------------
    # POLARIZATION UNCERTAINTY
    # -------------------------

    polarization_uncertain = (
        abs(
            polarization_calibrated_score
            - polarization_calibrated_threshold
        )
        < POLARIZATION_UNCERTAIN_DELTA
    )

    polarization_status = (
        "uncertain"
        if polarization_uncertain
        else "confident"
    )

    # =========================
    # DISPLAY ML RESULTS
    # =========================

    print(
        "Toxicity probability:",
        round(
            toxicity_score,
            4
        )
    )

    print(
        "Toxicity calibrated probability:",
        round(
            toxicity_calibrated_score,
            4
        )
    )

    print(
        "Toxicity status:",
        toxicity_status
    )

    print(
        "Toxicity:",
        toxicity_result
    )

    print(
        "Polarization probability:",
        round(
            polarization_score,
            4
        )
    )

    print(
        "Polarization calibrated probability:",
        round(
            polarization_calibrated_score,
            4
        )
    )

    print(
        "Polarization status:",
        polarization_status
    )

    print(
        "Polarization:",
        polarization_prediction
    )

    # =========================
    # MAFINDO RETRIEVAL
    # =========================

    print("\n=== MAFINDO RETRIEVAL ===")

    mafindo_results = search_mafindo(
        text
    )

    if not mafindo_results:

        print(
            "Tidak ditemukan reference MAfindo "
            "dengan similarity yang cukup."
        )

    else:

        print(
            "Reference ditemukan:",
            len(mafindo_results)
        )

        for i, item in enumerate(
            mafindo_results,
            start=1
        ):

            print(
                f"\n{i}.",
                item["title"]
            )

            print(
                "Similarity:",
                round(
                    item["score"],
                    4
                )
            )

            print(
                "Kategori dataset:",
                item["is_hoax"]
            )

    # =========================
    # BUILD REFERENCE CONTEXT
    # =========================

    reference_context = ""

    for i, item in enumerate(
        mafindo_results,
        start=1
    ):

        reference_context += f"""
REFERENCE {i}

Judul:
{item["title"]}

Kategori dataset:
{item["is_hoax"]}

Similarity:
{item["score"]:.4f}

Isi:
{item["content"]}
"""

    # =========================
    # GEMINI
    # =========================

    print("\n=== GEMINI ANALYSIS ===")

    prompt = f"""
Analisis teks berikut secara netral.

TEKS:
{text}

HASIL MODEL NLP SEBELUMNYA:

Toxicity:
- label: {toxicity_result}
- probability mentah: {toxicity_score:.4f}
- probability terkalibrasi: {toxicity_calibrated_score:.4f}
- status: {toxicity_status}

Polarization:
- label: {polarization_prediction}
- probability mentah: {polarization_score:.4f}
- probability terkalibrasi: {polarization_calibrated_score:.4f}
- status: {polarization_status}

Gunakan hasil model tersebut hanya sebagai sinyal tambahan,
bukan sebagai kebenaran mutlak.

PENTING TENTANG STATUS MODEL:
- Jika status toxicity adalah "uncertain", jangan menyatakan
  teks pasti toxic.
- Jika status polarization adalah "uncertain", jangan menyatakan
  teks pasti polarized.
- Probability model bukan probability bahwa isi klaim tersebut
  benar atau salah.
- Jangan mengubah hasil model menjadi klaim faktual tentang
  kebenaran suatu informasi.

Berikut adalah reference dari dataset MAfindo yang ditemukan
berdasarkan semantic similarity.

{
    reference_context
    if reference_context
    else
    "Tidak ada reference MAfindo yang memenuhi threshold."
}

PENTING TENTANG REFERENCE:
- Reference MAfindo hanya merupakan konteks pendukung.
- Similarity tinggi tidak berarti teks pengguna pasti sama
  dengan reference.
- Jangan menyatakan klaim pengguna pasti hoaks hanya karena
  reference memiliki kategori hoaks.
- Kategori dataset MAfindo bukan bukti otomatis bahwa klaim
  pengguna benar atau salah.
- Jika reference hanya menunjukkan pola yang mirip, jelaskan
  bahwa reference tersebut hanya relevan sebagai konteks.
- Jangan mengarang fakta, sumber, atau bukti yang tidak ada.

TUGAS:
1. Buat ringkasan singkat.
2. Identifikasi klaim utama jika ada.
3. Tentukan apakah isi utama berupa fact, opinion, prediction,
   atau mixed.
4. Jelaskan kemungkinan pola penalaran yang terlihat.
   Jangan menyatakan bahwa pola tersebut pasti merupakan
   logical fallacy.
5. Jelaskan bukti atau konteks apa yang dibutuhkan untuk
   memeriksa klaim tersebut.

Gunakan bahasa Indonesia yang netral dan hati-hati.
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
                "Gunakan bahasa yang hati-hati. "
                "Jangan menganggap prediksi model sebagai bukti "
                "bahwa suatu klaim benar atau salah."
            ),

            "response_mime_type": "application/json",

            "response_schema": ContextAnalysis,

            "temperature": 0
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
        "Calibrated probability:",
        round(
            toxicity_calibrated_score,
            4
        )
    )

    print(
        "Status:",
        toxicity_status
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
            polarization_score,
            4
        )
    )

    print(
        "Calibrated probability:",
        round(
            polarization_calibrated_score,
            4
        )
    )

    print(
        "Status:",
        polarization_status
    )

    print(
        "\n[MAfindo Reference]"
    )

    if mafindo_results:

        for i, item in enumerate(
            mafindo_results,
            start=1
        ):

            print(
                f"{i}. {item['title']} "
                f"(similarity={item['score']:.4f})"
            )

    else:

        print(
            "Tidak ada reference yang cukup relevan."
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

    # =========================
    # RETURN API RESULT
    # =========================

    return {
        "text": text,

        "toxicity": {
            "label": toxicity_result,
            "probability": toxicity_score,
            "calibrated_probability": (
                toxicity_calibrated_score
            ),
            "status": toxicity_status
        },

        "polarization": {
            "label": polarization_prediction,
            "probability": polarization_score,
            "calibrated_probability": (
                polarization_calibrated_score
            ),
            "status": polarization_status
        },

        "mafindo_references": [
            {
                "title": item["title"],
                "content": item["content"],
                "is_hoax": item["is_hoax"],
                "similarity": item["score"]
            }
            for item in mafindo_results
        ],

        "gemini": {
            "summary": result.summary,
            "claim": result.claim,
            "content_type": result.content_type,
            "reasoning_pattern": result.reasoning_pattern,
            "evidence_needed": result.evidence_needed
        }
    }
