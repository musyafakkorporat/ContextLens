import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "data/processed/dataset_preprocessed.csv"

df = pd.read_csv(input_file)


def split_dataset(df, label_column, name):
    # Ambil hanya data yang memiliki label
    data = df.dropna(subset=[label_column]).copy()

    # Pastikan label berupa integer
    data[label_column] = data[label_column].astype(int)

    # Split 70% train dan 30% sementara
    train, temp = train_test_split(
        data,
        test_size=0.30,
        random_state=42,
        stratify=data[label_column]
    )

    # Bagi 30% menjadi 15% validation dan 15% test
    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42,
        stratify=temp[label_column]
    )

    train.to_csv(
        f"data/processed/{name}_train.csv",
        index=False
    )

    validation.to_csv(
        f"data/processed/{name}_validation.csv",
        index=False
    )

    test.to_csv(
        f"data/processed/{name}_test.csv",
        index=False
    )

    print(f"\n=== {name.upper()} ===")
    print("Total     :", len(data))
    print("Train     :", len(train))
    print("Validation:", len(validation))
    print("Test      :", len(test))

    print("\nDistribusi Train:")
    print(train[label_column].value_counts(normalize=True).sort_index())

    print("\nDistribusi Validation:")
    print(validation[label_column].value_counts(normalize=True).sort_index())

    print("\nDistribusi Test:")
    print(test[label_column].value_counts(normalize=True).sort_index())


split_dataset(df, "toxicity_label", "toxicity")
split_dataset(df, "polarized_label", "polarized")