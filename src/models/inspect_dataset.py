
import pandas as pd

# Dataset path
file_path = "data/raw/candidate_dataset.csv.gz"

# Load compressed CSV
df = pd.read_csv(file_path, compression="gzip")

print("\n========== DATASET INFORMATION ==========")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset split distribution:")
print(df["dataset_split"].value_counts())

print("\nTarget distribution:")
print(df["target"].value_counts(dropna=False))

print("\nUnique trips:")
print(df["trip_id"].nunique())

print("\nUnique GPS observations:")
print(df[["trip_id", "observation_index"]].drop_duplicates().shape[0])

print("\nInspection completed!")
