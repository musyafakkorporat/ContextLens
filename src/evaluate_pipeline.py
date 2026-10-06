import os
import time

import numpy as np
import pandas as pd
import torch

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================
# CONFIG
# =========================

TEST_CASES_FILE = "data/test_cases.csv"

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

TOXICITY_THRESHOLD = 0.06

MAFINDO_MIN_SCORE = 0.62

MAFINDO_TOP_K = 5

MAFINDO_MAX_CONTEXT = 3

OUTPUT_FILE = "results/pipeline_evaluation_results.csv"


# =========================
# GEMINI OUTPUT
# =========================

class ContextAnalysis(BaseModel):

    summary: str
    claim: str
    content_type: str
    reasoning_pattern: str
    evidence_needed: str


# =========================
# LOAD ENV
# =========================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    raise ValueError(
        "GEMINI_API_KEY tidak ditemukan di .env"
    )

client = genai.Client(
    api_key=api_key
)


# =========================
# LOAD TEST CASES
# =========================

print("=== LOAD TEST CASES ===")

test_cases = pd.read_csv(
    TEST_CASES_FILE
)

print(
    "Jumlah test case:",
    len(test_cases)
)


# =========================
# LOAD MODELS
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


print("\n=== LOAD MAFINDO ===")

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
    "MAfindo:",
    len(mafindo_metadata),
    "data"
)


# =========================
# CLASSIFICATION
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

    return probabilities.numpy()


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

    references = []

    for index in top_indices:

        score = float(
            scores[index]
        )

        if score < MAFINDO_MIN_SCORE:
            continue

        row = mafindo_metadata.iloc[index]

        references.append({
            "title": str(row["title"]),
            "content": str(row["content"]),
            "is_hoax": str(row["is_hoax"]),
            "similarity": score
        })

    return references[:MAFINDO_MAX_CONTEXT]


# =========================
# GEMINI
# =========================

def analyze_with_gemini(
    text,
    references
):

    if references:

        reference_context = "\n\n".join(
            [
                (
                    f"REFERENSI {i + 1}\n"
                    f"Judul: {ref['title']}\n"
                    f"Kategori dataset: {ref['is_hoax']}\n"
                    f"Similarity: {ref['similarity']:.4f}\n"
                    f"Isi: {ref['content']}"
                )
                for i, ref in enumerate(references)
            ]
        )

    else:

        reference_context = (
            "Tidak ada referensi MAfindo "
            f"dengan similarity >= {MAFINDO_MIN_SCORE}."
        )

    prompt = f"""
Anda adalah modul analisis ContextLens.

Analisis teks berikut secara netral.

TEKS:
{text}

REFERENSI MAfindo:
{reference_context}

Aturan:

1. Tentukan content_type:
   fact, opinion, prediction, mixed, atau unclear.

2. Identifikasi klaim utama.

3. Buat ringkasan.

4. Identifikasi reasoning pattern jika memang terlihat.
   Jangan memaksakan adanya logical fallacy.

5. Jelaskan evidence yang diperlukan untuk memeriksa klaim.

6. Referensi MAfindo hanya konteks pendukung.
   Similarity tinggi bukan bukti bahwa teks pengguna benar
   atau hoaks.

7. Jangan mengarang fakta.

8. Gunakan bahasa Indonesia yang netral.

9. Jika terdapat politik atau pemerintah,
   jangan berpihak.
"""

    time.sleep(5)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ContextAnalysis
        }
    )

    return response.parsed


# =========================
# RUN TEST
# =========================

results = []

print("\n=== RUN EVALUATION ===")

for _, row in test_cases.iterrows():

    test_id = str(
        row["test_id"]
    )

    text = str(
        row["text"]
    )

    print(
        f"\n[{test_id}] {text}"
    )

    # -------------------------
    # Toxicity
    # -------------------------

    toxicity_prob = predict_class(
        text,
        toxicity_tokenizer,
        toxicity_model
    )

    toxicity_score = float(
        toxicity_prob[1]
    )

    toxicity_prediction = int(
        toxicity_score >= TOXICITY_THRESHOLD
    )

    # -------------------------
    # Polarization
    # -------------------------

    polarization_prob = predict_class(
        text,
        polarization_tokenizer,
        polarization_model
    )

    polarization_score = float(
        polarization_prob[1]
    )

    polarization_prediction = int(
        np.argmax(polarization_prob)
    )

    # -------------------------
    # MAfindo
    # -------------------------

    references = search_mafindo(
        text
    )

    reference_found = int(
        len(references) > 0
    )

    top_similarity = (
        references[0]["similarity"]
        if references
        else None
    )

    best_reference = (
        references[0]["title"]
        if references
        else ""
    )

    # -------------------------
    # Gemini
    # -------------------------

    print(
        "Menjalankan Gemini..."
    )

    try:

        analysis = analyze_with_gemini(
            text,
            references
        )

        summary = analysis.summary
        claim = analysis.claim
        content_type = analysis.content_type
        reasoning_pattern = (
            analysis.reasoning_pattern
        )
        evidence_needed = (
            analysis.evidence_needed
        )

    except Exception as e:

        print(
            "Gemini error:",
            str(e)
        )

        summary = ""
        claim = ""
        content_type = "error"
        reasoning_pattern = ""
        evidence_needed = ""


    results.append({

        "test_id": test_id,

        "text": text,

        "expected_toxicity":
            int(row["expected_toxicity"]),

        "predicted_toxicity":
            toxicity_prediction,

        "toxicity_probability":
            toxicity_score,

        "expected_polarization":
            int(row["expected_polarization"]),

        "predicted_polarization":
            polarization_prediction,

        "polarization_probability":
            polarization_score,

        "expected_content_type":
            str(row["expected_content_type"]),

        "predicted_content_type":
            content_type,

        "expected_reference":
            str(row["expected_reference"]),

        "reference_found":
            reference_found,

        "top_similarity":
            top_similarity,

        "best_reference":
            best_reference,

        "summary":
            summary,

        "claim":
            claim,

        "reasoning_pattern":
            reasoning_pattern,

        "evidence_needed":
            evidence_needed
    })


# =========================
# DATAFRAME
# =========================

results_df = pd.DataFrame(
    results
)

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================
# TOXICITY METRICS
# =========================

y_true_toxicity = (
    results_df["expected_toxicity"]
)

y_pred_toxicity = (
    results_df["predicted_toxicity"]
)

print("\n")
print("=" * 70)
print("TOXICITY EVALUATION")
print("=" * 70)

print(
    "Accuracy :",
    round(
        accuracy_score(
            y_true_toxicity,
            y_pred_toxicity
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_true_toxicity,
            y_pred_toxicity,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall   :",
    round(
        recall_score(
            y_true_toxicity,
            y_pred_toxicity,
            zero_division=0
        ),
        4
    )
)

print(
    "F1       :",
    round(
        f1_score(
            y_true_toxicity,
            y_pred_toxicity,
            zero_division=0
        ),
        4
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_true_toxicity,
        y_pred_toxicity
    )
)


# =========================
# POLARIZATION METRICS
# =========================

y_true_polarization = (
    results_df["expected_polarization"]
)

y_pred_polarization = (
    results_df["predicted_polarization"]
)

print("\n")
print("=" * 70)
print("POLARIZATION EVALUATION")
print("=" * 70)

print(
    "Accuracy :",
    round(
        accuracy_score(
            y_true_polarization,
            y_pred_polarization
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_true_polarization,
            y_pred_polarization,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall   :",
    round(
        recall_score(
            y_true_polarization,
            y_pred_polarization,
            zero_division=0
        ),
        4
    )
)

print(
    "F1       :",
    round(
        f1_score(
            y_true_polarization,
            y_pred_polarization,
            zero_division=0
        ),
        4
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_true_polarization,
        y_pred_polarization
    )
)


# =========================
# CONTENT TYPE
# =========================

# =========================
# CONTENT TYPE
# =========================

valid_content_results = results_df[
    results_df["predicted_content_type"] != "error"
].copy()

print("\n")
print("=" * 70)
print("CONTENT TYPE EVALUATION")
print("=" * 70)

if len(valid_content_results) == 0:

    print("Tidak ada hasil Gemini yang berhasil.")
    print("Content type tidak dapat dievaluasi.")

else:

    content_accuracy = accuracy_score(
        valid_content_results["expected_content_type"],
        valid_content_results["predicted_content_type"]
    )

    print(
        "Accuracy:",
        round(
            content_accuracy,
            4
        )
    )

    print(
        "Jumlah evaluasi berhasil:",
        len(valid_content_results)
    )

    print(
        "Jumlah error Gemini:",
        len(results_df) - len(valid_content_results)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            valid_content_results["expected_content_type"],
            valid_content_results["predicted_content_type"],
            zero_division=0
        )
    )

# =========================
# MAFINDO RETRIEVAL
# =========================

expected_reference = (
    results_df["expected_reference"]
    .str.lower()
)

predicted_reference = (
    results_df["reference_found"]
)

expected_reference_binary = (
    expected_reference == "yes"
).astype(int)

print("\n")
print("=" * 70)
print("MAFINDO RETRIEVAL EVALUATION")
print("=" * 70)

print(
    "Accuracy:",
    round(
        accuracy_score(
            expected_reference_binary,
            predicted_reference
        ),
        4
    )
)

print(
    "Expected reference:",
    int(
        expected_reference_binary.sum()
    )
)

print(
    "Reference ditemukan:",
    int(
        predicted_reference.sum()
    )
)


# =========================
# FINISH
# =========================

print("\n")
print("=" * 70)
print("EVALUATION SELESAI")
print("=" * 70)

print(
    "Jumlah test case:",
    len(results_df)
)

print(
    "Hasil disimpan di:",
    OUTPUT_FILE
)