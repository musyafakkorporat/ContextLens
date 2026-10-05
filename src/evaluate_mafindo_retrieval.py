import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


EMBEDDINGS_FILE = "data/reference/mafindo_embeddings.npy"
METADATA_FILE = "data/reference/mafindo_metadata.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 5
MIN_SCORE = 0.62
MAX_CONTEXT = 3


QUERIES = [
    "token listrik gratis 500 ribu untuk pengguna Telegram",
    "hadiah voucher pulsa Telegram 500 ribu",
    "akun Telegram yang menawarkan investasi Modalku",
    "SMS Telkomsel hadiah 100 juta",
    "bantuan beras gratis 30 kilogram",
    "investasi online yang mengatasnamakan OJK",
]


def search(query, model, embeddings, metadata):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for index in top_indices:
        results.append({
            "score": float(scores[index]),
            "title": str(metadata.iloc[index]["title"]),
            "is_hoax": int(metadata.iloc[index]["is_hoax"])
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
    print("=" * 90)
    print("EVALUASI SEMANTIC RETRIEVAL MAfindo")
    print("=" * 90)
    print(f"Top K             : {TOP_K}")
    print(f"Minimal similarity: {MIN_SCORE}")
    print(f"Maksimal context  : {MAX_CONTEXT}")

    for query_number, query in enumerate(QUERIES, start=1):

        print()
        print("=" * 90)
        print(f"QUERY {query_number}")
        print("=" * 90)
        print(query)

        results = search(
            query,
            model,
            embeddings,
            metadata
        )

        print()
        print("Top hasil:")

        for rank, result in enumerate(results, start=1):
            category = (
                "Hoaks"
                if result["is_hoax"] == 1
                else "Non-hoaks"
            )

            print(
                f"{rank}. "
                f"Score={result['score']:.4f} | "
                f"{category} | "
                f"{result['title']}"
            )

        filtered = [
            result
            for result in results
            if result["score"] >= MIN_SCORE
        ]

        filtered = filtered[:MAX_CONTEXT]

        print()
        print(
            f"Setelah filter >= {MIN_SCORE}: "
            f"{len(filtered)} reference"
        )

        for rank, result in enumerate(filtered, start=1):
            print(
                f"  {rank}. "
                f"{result['score']:.4f} | "
                f"{result['title']}"
            )

    print()
    print("=" * 90)
    print("EVALUASI SELESAI")
    print("=" * 90)


if __name__ == "__main__":
    main()