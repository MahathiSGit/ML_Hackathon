
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "candidate_dataset.csv.gz"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_dataset():

    print("\n========== LOADING DATASET ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, compression="gzip")

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)

    return df


# ==========================================
# 3. BASIC DATASET INFORMATION
# ==========================================

def basic_information(df):

    print("\n========== BASIC INFORMATION ==========")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nDataset information:")
    df.info()


# ==========================================
# 4. MISSING VALUE ANALYSIS
# ==========================================

def missing_value_analysis(df):

    print("\n========== MISSING VALUE ANALYSIS ==========")

    missing = pd.DataFrame({
        "Missing_Count": df.isnull().sum(),
        "Missing_Percentage": df.isnull().mean() * 100
    })

    missing = missing.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    print(missing)

    missing.to_csv(OUTPUT_DIR / "missing_values.csv")

    print("\nMissing value report saved!")


# ==========================================
# 5. DUPLICATE ANALYSIS
# ==========================================

def duplicate_analysis(df):

    print("\n========== DUPLICATE ANALYSIS ==========")

    duplicate_count = df.duplicated().sum()

    print("Duplicate rows:", duplicate_count)

    # Do not remove duplicates during EDA.


# ==========================================
# 6. NUMERICAL FEATURE ANALYSIS
# ==========================================

def numerical_analysis(df):

    print("\n========== NUMERICAL FEATURE ANALYSIS ==========")

    numerical_df = df.select_dtypes(include=np.number)

    summary = numerical_df.describe().T

    print(summary)

    summary.to_csv(OUTPUT_DIR / "numerical_summary.csv")

    print("\nNumerical summary saved!")

    # Histograms
    numerical_df.hist(
        bins=30,
        figsize=(15, 12),
        edgecolor="black"
    )

    plt.suptitle("Numerical Feature Distributions")
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "numerical_distributions.png")
    plt.close()


# ==========================================
# 7. TARGET DISTRIBUTION
# ==========================================

def target_analysis(df):

    print("\n========== TARGET DISTRIBUTION ==========")

    target_counts = df["target"].value_counts().sort_index()

    print(target_counts)

    print("\nTarget percentages:")
    print(df["target"].value_counts(normalize=True) * 100)

    target_counts.plot(
        kind="bar",
        figsize=(7, 5),
        edgecolor="black"
    )

    plt.title("Target Class Distribution")
    plt.xlabel("Target (0 = Incorrect, 1 = Correct)")
    plt.ylabel("Number of Candidates")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "target_distribution.png")
    plt.close()

    target_counts.to_csv(OUTPUT_DIR / "target_distribution.csv")


# ==========================================
# 8. DATASET SPLIT ANALYSIS
# ==========================================

def split_analysis(df):

    print("\n========== DATASET SPLIT ANALYSIS ==========")

    print("\nCandidate rows per split:")
    print(df["dataset_split"].value_counts())

    print("\nTarget distribution per split:")

    split_target = pd.crosstab(
        df["dataset_split"],
        df["target"]
    )

    print(split_target)

    split_target.to_csv(OUTPUT_DIR / "split_target_distribution.csv")


# ==========================================
# 9. CANDIDATES PER OBSERVATION
# ==========================================

def candidate_analysis(df):

    print("\n========== CANDIDATE ANALYSIS ==========")

    observation_columns = ["trip_id", "observation_index"]

    candidates = (
        df.groupby(observation_columns)
        .size()
        .rename("candidate_count")
    )

    print("\nCandidate count statistics:")
    print(candidates.describe())

    candidates.to_csv(
        OUTPUT_DIR / "candidates_per_observation.csv"
    )

    plt.figure(figsize=(9, 5))

    plt.hist(
        candidates,
        bins=40,
        edgecolor="black"
    )

    plt.title("Candidates Per GPS Observation")
    plt.xlabel("Number of Candidate Roads")
    plt.ylabel("Number of Observations")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "candidate_distribution.png")
    plt.close()


# ==========================================
# 10. FEATURE CORRELATION ANALYSIS
# ==========================================

def correlation_analysis(df):

    print("\n========== FEATURE CORRELATION ==========")

    feature_columns = [
        "candidate_rank",
        "distance_m",
        "heading_difference_deg",
        "road_length_m",
        "gps_speed_mps",
        "target"
    ]

    correlation = df[feature_columns].corr()

    print(correlation)

    correlation.to_csv(OUTPUT_DIR / "feature_correlation.csv")

    plt.figure(figsize=(9, 7))

    plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(feature_columns)),
        feature_columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(feature_columns)),
        feature_columns
    )

    for i in range(len(feature_columns)):
        for j in range(len(feature_columns)):
            plt.text(
                j, i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "correlation_heatmap.png")
    plt.close()


# ==========================================
# 11. OUTLIER ANALYSIS
# ==========================================

def outlier_analysis(df):

    print("\n========== OUTLIER ANALYSIS ==========")

    feature_columns = [
        "candidate_rank",
        "distance_m",
        "heading_difference_deg",
        "road_length_m",
        "gps_speed_mps"
    ]

    results = []

    for column in feature_columns:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ).sum()

        results.append({
            "Feature": column,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outlier_Count": outliers,
            "Outlier_Percentage": outliers / len(df) * 100
        })

    outlier_df = pd.DataFrame(results)

    print(outlier_df)

    outlier_df.to_csv(
        OUTPUT_DIR / "outlier_analysis.csv",
        index=False
    )


# ==========================================
# 12. RUN ALL EDA FUNCTIONS
# ==========================================

def main():

    print("\n========================================")
    print("       PATHSNAP - EXPLORATORY DATA ANALYSIS")
    print("========================================")

    df = load_dataset()

    basic_information(df)

    missing_value_analysis(df)

    duplicate_analysis(df)

    numerical_analysis(df)

    target_analysis(df)

    split_analysis(df)

    candidate_analysis(df)

    correlation_analysis(df)

    outlier_analysis(df)

    print("\n========================================")
    print("EDA COMPLETED SUCCESSFULLY!")
    print("Reports saved in:", OUTPUT_DIR)
    print("Plots saved in:", PLOT_DIR)
    print("========================================")


if __name__ == "__main__":
    main()
