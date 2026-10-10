import pandas as pd

file_path = "data/raw/indotoxic2024_annotated_data_v2_final.csv"

df = pd.read_csv(file_path)

print("=== INFORMASI DATASET ===")
print("Jumlah baris :", len(df))
print("Jumlah kolom :", len(df.columns))

print("\n=== NAMA KOLOM ===")
print(df.columns.tolist())

print("\n=== 5 DATA PERTAMA ===")
print(df.head())

print("\n=== TIPE DATA ===")
print(df.dtypes)

print("\n=== DATA KOSONG ===")
print(df.isnull().sum())

print("\n=== JUMLAH DATA DUPLIKAT ===")
print(df.duplicated().sum())