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
MAX_CONTEXT = 3
MIN_SCORE = 0.62


def load_retriever():
    print("Load embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)

    print("Load metadata...")
    metadata = pd.read_csv(METADATA_FILE)

    print("Load embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    return model, embeddings, metadata


def semantic_search(query, model, embeddings, metadata):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for index in top_indices:
        score = float(scores[index])

        results.append({
            "score": score,
            "title": str(metadata.iloc[index]["title"]),
            "content": str(metadata.iloc[index]["content"]),
            "is_hoax": int(metadata.iloc[index]["is_hoax"])
        })

    return results


def filter_results(results):
    filtered = [
        result
        for result in results
        if result["score"] >= MIN_SCORE
    ]

    return filtered[:MAX_CONTEXT]


def build_context(results):
    context_parts = []

    for i, result in enumerate(results, start=1):
        category = (
            "Hoaks"
            if result["is_hoax"] == 1
            else "Non-hoaks"
        )

        context_parts.append(
            f"""
REFERENSI {i}
Judul: {result['title']}
Kategori pada dataset: {category}
Similarity score: {result['score']:.4f}

Isi:
{result['content']}
"""
        )

    return "\n".join(context_parts)


def analyze_with_gemini(query, context, client):
    prompt = f"""
Anda adalah bagian dari sistem ContextLens.

Tugas Anda adalah membantu pengguna memahami informasi
berdasarkan referensi yang ditemukan.

Teks pengguna:
{query}

Referensi:
{context}

Aturan penting:
1. Referensi hanya digunakan sebagai konteks pendukung.
2. Jangan menyatakan teks pengguna pasti benar atau pasti hoaks
   hanya berdasarkan similarity.
3. Jangan menganggap kategori dataset sebagai kebenaran otomatis
   terhadap teks pengguna.
4. Jelaskan apabila referensi hanya menunjukkan pola yang mirip.
5. Jangan mengarang fakta yang tidak terdapat pada referensi.
6. Gunakan bahasa Indonesia yang sederhana dan netral.

Berikan:

Ringkasan:
- Ringkas informasi pengguna.

Kecocokan dengan referensi:
- Jelaskan referensi mana yang paling relevan.
- Jelaskan persamaan konteksnya.

Konteks:
- Jelaskan informasi penting dari referensi.

Kesimpulan:
- Berikan kesimpulan yang hati-hati.
- Jika bukti belum cukup, katakan bahwa bukti belum cukup.
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
    print("=" * 80)
    print(f"Minimal similarity : {MIN_SCORE}")
    print(f"Maksimal context   : {MAX_CONTEXT}")
    print("Ketik 'exit' untuk keluar.")

    while True:
        query = input("\nTeks pengguna: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Teks tidak boleh kosong.")
            continue

        print("\nMencari referensi...")

        all_results = semantic_search(
            query,
            model,
            embeddings,
            metadata
        )

        print("\nTop hasil sebelum filter:")

        for i, result in enumerate(all_results, start=1):
            print(
                f"{i}. {result['title']} "
                f"(score={result['score']:.4f})"
            )

        results = filter_results(all_results)

        print(
            f"\nReferensi setelah filter: "
            f"{len(results)}"
        )

        if not results:
            print(
                "Tidak ditemukan referensi dengan "
                f"similarity >= {MIN_SCORE}."
            )
            continue

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