import pandas as pd

file_path = "data/raw/indotoxic2024_annotated_data_v2_final.csv"

df = pd.read_csv(file_path)

print("=== CONTOH LABEL TOXICITY ===")
print(df["toxicity"].value_counts().head(20))

print("\n=== CONTOH LABEL POLARIZED ===")
print(df["polarized"].value_counts().head(20))

print("\n=== CONTOH ISI LABEL ===")
for i in range(10):
    print("Teks ID:", df.loc[i, "text_id"])
    print("Toxicity:", df.loc[i, "toxicity"])
    print("Polarized:", df.loc[i, "polarized"])
    print()