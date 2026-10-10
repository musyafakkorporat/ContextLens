import pandas as pd

file_path = "data/raw/indotoxic2024_annotated_data_v2_final.csv"

df = pd.read_csv(file_path)

print("=== PEMERIKSAAN TEXT ===")

print("Jumlah text kosong:")
print(df["text"].isnull().sum())

print("\nJumlah text kosong setelah strip:")
print(df["text"].fillna("").str.strip().eq("").sum())

print("\nPanjang text:")
print(df["text"].fillna("").str.len().describe())