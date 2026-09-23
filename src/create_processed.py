import pandas as pd
import ast
from collections import Counter

input_file = "data/raw/indotoxic2024_annotated_data_v2_final.csv"
output_file = "data/processed/dataset_clean.csv"

df = pd.read_csv(input_file)


def majority_vote(label_string):
    if pd.isna(label_string):
        return None

    labels = ast.literal_eval(label_string)
    labels = [int(label) for label in labels]

    count = Counter(labels)

    if count[0] > count[1]:
        return 0
    elif count[1] > count[0]:
        return 1
    else:
        return None


df["toxicity_label"] = df["toxicity"].apply(majority_vote)
df["polarized_label"] = df["polarized"].apply(majority_vote)

# Hapus data yang tidak memiliki teks
df = df[df["text"].notna()].copy()

# Ambil kolom yang diperlukan
df = df[
    [
        "text_id",
        "text",
        "topic",
        "toxicity_label",
        "polarized_label"
    ]
]

# Simpan dataset processed
df.to_csv(output_file, index=False)

print("=== DATASET PROCESSED ===")
print("Jumlah data:", len(df))

print("\n=== TOXICITY ===")
print(df["toxicity_label"].value_counts(dropna=False).sort_index())

print("\n=== POLARIZED ===")
print(df["polarized_label"].value_counts(dropna=False).sort_index())

print("\nDataset disimpan ke:")
print(output_file)