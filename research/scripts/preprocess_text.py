import pandas as pd
import re

input_file = "data/processed/dataset_clean.csv"
output_file = "data/processed/dataset_preprocessed.csv"

df = pd.read_csv(input_file)


def clean_text(text):
    text = str(text)

    # Ubah menjadi huruf kecil
    text = text.lower()

    # Rapikan whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["text_clean"] = df["text"].apply(clean_text)

df.to_csv(output_file, index=False)

print("=== PREPROCESSING SELESAI ===")
print("Jumlah data:", len(df))

print("\n=== CONTOH SEBELUM DAN SESUDAH ===")

for i in range(5):
    print("\nID:", df.loc[i, "text_id"])
    print("Sebelum :", df.loc[i, "text"])
    print("Sesudah :", df.loc[i, "text_clean"])

print("\nFile disimpan ke:")
print(output_file)