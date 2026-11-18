import pandas as pd

df = pd.read_csv(
    "data/raw/candidate_dataset.csv.gz",
    compression="gzip"
)

print("\n========== DATA VALIDATION ==========")

# 1. Check correct candidates per GPS observation
positive_counts = (
    df.groupby(["trip_id", "observation_index"])["target"].sum()
)

print("\nCorrect candidates per observation:")
print(positive_counts.value_counts().sort_index())

# 2. Check whether trips appear in multiple splits
trip_splits = df.groupby("trip_id")["dataset_split"].nunique()

print("\nTrips appearing in multiple splits:")
print((trip_splits > 1).sum())

# 3. Number of unique trips in each split
print("\nUnique trips per split:")
print(df.groupby("dataset_split")["trip_id"].nunique())

# 4. Unique GPS observations per split
observations = (
    df.groupby("dataset_split")
    .apply(
        lambda x: x[["trip_id", "observation_index"]]
        .drop_duplicates()
        .shape[0]
    )
)

print("\nUnique observations per split:")
print(observations)

# 5. Target distribution in each split
print("\nTarget distribution per split:")
print(pd.crosstab(df["dataset_split"], df["target"]))

# 6. Candidates per GPS observation
candidate_counts = (
    df.groupby(["trip_id", "observation_index"]).size()
)

print("\nCandidates per observation:")
print(candidate_counts.describe())

print("\nValidation completed!")