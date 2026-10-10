import pandas as pd

file_path = "data/processed/dataset_clean.csv"

df = pd.read_csv(file_path)

print("=== INFORMASI DATASET PROCESSED ===")
print("Jumlah baris :", len(df))
print("Jumlah kolom :", len(df.columns))

print("\n=== NAMA KOLOM ===")
print(df.columns.tolist())

print("\n=== 5 DATA PERTAMA ===")
print(df.head())

print("\n=== DATA KOSONG ===")
print(df.isnull().sum())

print("\n=== DISTRIBUSI TOXICITY ===")
print(df["toxicity_label"].value_counts(dropna=False))

print("\n=== DISTRIBUSI POLARIZED ===")
print(df["polarized_label"].value_counts(dropna=False))