import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/reference/mafindo_2024_clean.csv"
OUTPUT_EMBEDDINGS = "data/reference/mafindo_embeddings.npy"
OUTPUT_METADATA = "data/reference/mafindo_metadata.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    print("Load dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("Jumlah data:", len(df))

    print("Load embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = df["search_text"].fillna("").tolist()

    print("Membuat embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    os.makedirs("data/reference", exist_ok=True)

    np.save(OUTPUT_EMBEDDINGS, embeddings)

    metadata = df[
        ["title", "content", "is_hoax"]
    ].copy()

    metadata.to_csv(
        OUTPUT_METADATA,
        index=False
    )

    print()
    print("Embedding selesai.")
    print("Shape:", embeddings.shape)
    print("Embedding:", OUTPUT_EMBEDDINGS)
    print("Metadata:", OUTPUT_METADATA)


if __name__ == "__main__":
    main()