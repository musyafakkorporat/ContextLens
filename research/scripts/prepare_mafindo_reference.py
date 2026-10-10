import os
import pandas as pd


INPUT_FILE = "data/reference/mafindo_2024.csv"
OUTPUT_FILE = "data/reference/mafindo_2024_clean.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Data awal:", len(df))

    # Hapus data duplikat berdasarkan title + content
    df = df.drop_duplicates(subset=["title", "content"])

    # Pastikan text tidak kosong
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["content"] = df["content"].fillna("").astype(str).str.strip()

    # Hapus content yang terlalu pendek
    df = df[df["content"].str.len() >= 50].copy()

    # Gabungkan judul dan isi untuk semantic search
    df["search_text"] = (
        "Judul: " + df["title"] +
        "\nIsi: " + df["content"]
    )

    # Reset index
    df = df.reset_index(drop=True)

    os.makedirs("data/reference", exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Data setelah cleaning:", len(df))
    print("Kolom:", df.columns.tolist())
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()