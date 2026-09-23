import pandas as pd
import ast
from collections import Counter

file_path = "data/raw/indotoxic2024_annotated_data_v2_final.csv"

df = pd.read_csv(file_path)


def majority_vote(label_string):
    labels = ast.literal_eval(label_string)
    labels = [int(label) for label in labels]

    count = Counter(labels)

    if count[0] > count[1]:
        return 0
    elif count[1] > count[0]:
        return 1
    else:
        return -1


df["toxicity_final"] = df["toxicity"].apply(majority_vote)
df["polarized_final"] = df["polarized"].apply(majority_vote)


print("=== HASIL MAJORITY VOTE TOXICITY ===")
print(df["toxicity_final"].value_counts().sort_index())

print("\n=== HASIL MAJORITY VOTE POLARIZED ===")
print(df["polarized_final"].value_counts().sort_index())

print("\n=== DATA DENGAN HASIL SERI ===")
print("Toxicity :", (df["toxicity_final"] == -1).sum())
print("Polarized:", (df["polarized_final"] == -1).sum())

print("\n=== CONTOH DATA TOXICITY SERI ===")

tie_toxicity = df[df["toxicity_final"] == -1]

for i in range(min(10, len(tie_toxicity))):
    row = tie_toxicity.iloc[i]

    print("Teks ID:", row["text_id"])
    print("Teks:", row["text"])
    print("Annotators:", row["annotators_id"])
    print("Toxicity:", row["toxicity"])
    print()


print("\n=== CONTOH DATA POLARIZED SERI ===")

tie_polarized = df[df["polarized_final"] == -1]

for i in range(min(10, len(tie_polarized))):
    row = tie_polarized.iloc[i]

    print("Teks ID:", row["text_id"])
    print("Teks:", row["text"])
    print("Annotators:", row["annotators_id"])
    print("Polarized:", row["polarized"])
    print()