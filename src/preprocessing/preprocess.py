
import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "candidate_dataset.csv.gz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset columns
FEATURE_COLUMNS = [
    "candidate_rank",
    "distance_m",
    "heading_difference_deg",
    "road_length_m",
    "gps_speed_mps"
]

ID_COLUMNS = [
    "trip_id",
    "observation_index",
    "candidate_arc_id"
]

TARGET_COLUMN = "target"
SPLIT_COLUMN = "dataset_split"


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, compression="gzip")

    print("Dataset shape:", df.shape)

    # Separate dataset splits
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    val_df = df[df[SPLIT_COLUMN] == "validation"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()

    # Keep student_test_dev separate
    dev_df = df[df[SPLIT_COLUMN] == "student_test_dev"].copy()

    print("\nSplit sizes:")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)
    print("Student test dev:", dev_df.shape)

    # Fit imputer ONLY on training features
    print("\nFitting median imputer on training data...")

    imputer = SimpleImputer(strategy="median")

    train_df[FEATURE_COLUMNS] = imputer.fit_transform(
        train_df[FEATURE_COLUMNS]
    )

    # Apply the same fitted imputer to other splits
    val_df[FEATURE_COLUMNS] = imputer.transform(
        val_df[FEATURE_COLUMNS]
    )

    test_df[FEATURE_COLUMNS] = imputer.transform(
        test_df[FEATURE_COLUMNS]
    )

    # Transform student_test_dev without fitting on it
    dev_df[FEATURE_COLUMNS] = imputer.transform(
        dev_df[FEATURE_COLUMNS]
    )

    # Save processed splits
    print("\nSaving processed datasets...")

    train_df.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val_df.to_parquet(OUTPUT_DIR / "validation.parquet", index=False)
    test_df.to_parquet(OUTPUT_DIR / "test.parquet", index=False)
    dev_df.to_parquet(OUTPUT_DIR / "student_test_dev.parquet", index=False)

    # Save fitted imputer
    joblib.dump(
        imputer,
        ARTIFACT_DIR / "median_imputer.joblib"
    )

    print("\nPreprocessing completed successfully!")

    print("\nRemaining missing values in training features:")
    print(train_df[FEATURE_COLUMNS].isnull().sum())

    print("\nRemaining missing values in validation features:")
    print(val_df[FEATURE_COLUMNS].isnull().sum())

    print("\nRemaining missing values in test features:")
    print(test_df[FEATURE_COLUMNS].isnull().sum())

    print("\nSaved files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
