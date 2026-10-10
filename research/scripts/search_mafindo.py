import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


EMBEDDINGS_FILE = "data/reference/mafindo_embeddings.npy"
METADATA_FILE = "data/reference/mafindo_metadata.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 5


def search(query, model, embeddings, metadata, top_k=5):
    # Ubah query menjadi embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Karena embedding sudah dinormalisasi,
    # dot product = cosine similarity
    scores = embeddings @ query_embedding

    # Ambil index dengan score tertinggi
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "score": float(scores[index]),
            "title": metadata.iloc[index]["title"],
            "is_hoax": metadata.iloc[index]["is_hoax"],
            "content": metadata.iloc[index]["content"]
        })

    return results


def main():
    print("Load embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)

    print("Load metadata...")
    metadata = pd.read_csv(METADATA_FILE)

    print("Load embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print()
    print("Semantic Search MAfindo")
    print("Ketik 'exit' untuk keluar.")
    print()

    while True:
        query = input("Query: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Query tidak boleh kosong.\n")
            continue

        results = search(
            query,
            model,
            embeddings,
            metadata,
            TOP_K
        )

        print()
        print("=" * 80)

        for i, result in enumerate(results, start=1):
            print(f"\n[{i}] Score: {result['score']:.4f}")
            print(f"Judul    : {result['title']}")
            print(f"is_hoax  : {result['is_hoax']}")
            print(f"Isi      : {result['content'][:500]}...")

        print()
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()