import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


EMBEDDINGS_FILE = "data/reference/mafindo_embeddings.npy"
METADATA_FILE = "data/reference/mafindo_metadata.csv"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-3.5-flash-lite"

TOP_K = 5


def load_retriever():
    print("Load embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)

    print("Load metadata...")
    metadata = pd.read_csv(METADATA_FILE)

    print("Load embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    return model, embeddings, metadata


def semantic_search(query, model, embeddings, metadata, top_k=5):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "score": float(scores[index]),
            "title": str(metadata.iloc[index]["title"]),
            "content": str(metadata.iloc[index]["content"]),
            "is_hoax": int(metadata.iloc[index]["is_hoax"])
        })

    return results


def build_context(results):
    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
REFERENSI {i}
Judul: {result['title']}
Kategori referensi: {"Hoaks" if result['is_hoax'] == 1 else "Non-hoaks"}
Kemiripan: {result['score']:.4f}

Isi:
{result['content']}
"""
        )

    return "\n".join(context_parts)


def analyze_with_gemini(query, context, client):
    prompt = f"""
Anda adalah bagian dari sistem ContextLens.

Tugas Anda adalah membantu pengguna memahami sebuah informasi
dengan menggunakan referensi yang ditemukan dari database MAfindo.

Jangan langsung menyatakan bahwa teks pengguna benar atau hoaks
hanya berdasarkan kemiripan dengan referensi.

Gunakan bahasa Indonesia yang sederhana dan netral.

Teks pengguna:
{query}

Referensi yang ditemukan:
{context}

Berikan analisis dengan format:

Ringkasan:
- Jelaskan secara singkat isi informasi pengguna.

Kecocokan dengan referensi:
- Jelaskan apakah terdapat referensi yang memiliki konteks atau
  pola yang mirip.
- Sebutkan referensi yang paling relevan.

Konteks:
- Jelaskan informasi penting yang ditemukan dari referensi.

Kesimpulan:
- Berikan kesimpulan yang hati-hati.
- Jika referensi menunjukkan klaim serupa pernah dikategorikan
  sebagai hoaks, katakan bahwa "referensi tersebut dikategorikan
  sebagai hoaks", bukan bahwa teks pengguna pasti hoaks.
- Jika tidak cukup bukti, katakan bahwa informasi belum cukup
  untuk menentukan kebenaran klaim.

Jangan mengarang fakta yang tidak terdapat pada referensi.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY tidak ditemukan di file .env"
        )

    client = genai.Client(api_key=api_key)

    model, embeddings, metadata = load_retriever()

    print()
    print("=" * 80)
    print("ContextLens - RAG MAfindo")
    print("Ketik 'exit' untuk keluar.")
    print("=" * 80)

    while True:
        query = input("\nTeks pengguna: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Teks tidak boleh kosong.")
            continue

        print("\nMencari referensi...")
        results = semantic_search(
            query,
            model,
            embeddings,
            metadata,
            TOP_K
        )

        print("\nReferensi ditemukan:")

        for i, result in enumerate(results, start=1):
            print(
                f"{i}. {result['title']} "
                f"(score={result['score']:.4f})"
            )

        print("\nMembangun context...")
        context = build_context(results)

        print("Mengirim context ke Gemini...")

        answer = analyze_with_gemini(
            query,
            context,
            client
        )

        print("\n")
        print("=" * 80)
        print("HASIL ANALISIS")
        print("=" * 80)
        print(answer)
        print("=" * 80)


if __name__ == "__main__":
    main()