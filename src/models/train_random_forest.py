
import pandas as pd
import joblib
import json
import time

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "validation.parquet"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
REPORT_PATH = REPORT_DIR / "random_forest_metrics.json"

# --------------------------------------------------
# 2. Feature and target columns
# --------------------------------------------------

FEATURE_COLUMNS = [
    "candidate_rank",
    "distance_m",
    "heading_difference_deg",
    "road_length_m",
    "gps_speed_mps"
]

TARGET_COLUMN = "target"

ID_COLUMNS = [
    "trip_id",
    "observation_index",
    "candidate_arc_id"
]


def main():

    start_time = time.time()

    # --------------------------------------------------
    # 3. Load processed datasets
    # --------------------------------------------------

    print("Loading processed datasets...")

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    print("Training shape:", train_df.shape)
    print("Validation shape:", val_df.shape)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    # --------------------------------------------------
    # 4. Initialize Random Forest
    # --------------------------------------------------

    print("\nInitializing Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=18,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        max_samples=0.3,
        class_weight="balanced_subsample",
        bootstrap=True,
        n_jobs=4,
        random_state=42,
        verbose=1
    )

    # --------------------------------------------------
    # 5. Train model
    # --------------------------------------------------

    print("\nStarting Random Forest training...")

    rf_model.fit(X_train, y_train)

    print("\nTraining completed!")

    # --------------------------------------------------
    # 6. Make validation predictions
    # --------------------------------------------------

    print("\nEvaluating on validation data...")

    y_pred = rf_model.predict(X_val)
    y_prob = rf_model.predict_proba(X_val)[:, 1]

    # --------------------------------------------------
    # 7. Calculate classification metrics
    # --------------------------------------------------

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(
            precision_score(y_val, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_val, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_val, y_prob)
        ),
        "pr_auc": float(
            average_precision_score(y_val, y_prob)
        )
    }

    # --------------------------------------------------
    # 8. PathSnap top-1 candidate accuracy
    # --------------------------------------------------

    print("\nCalculating top-1 candidate accuracy...")

    # FIX: Include candidate_rank in the selected columns
    ranking_df = val_df[
        ID_COLUMNS + ["candidate_rank", TARGET_COLUMN]
    ].copy()

    ranking_df["predicted_probability"] = y_prob

    ranking_df = ranking_df.sort_values(
        [
            "trip_id",
            "observation_index",
            "predicted_probability",
            "candidate_rank"
        ],
        ascending=[True, True, False, True]
    )

    best_candidates = ranking_df.drop_duplicates(
        subset=["trip_id", "observation_index"],
        keep="first"
    )

    top1_accuracy = best_candidates[TARGET_COLUMN].mean()

    metrics["top1_candidate_accuracy"] = float(top1_accuracy)

    # --------------------------------------------------
    # 9. Print results
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("RANDOM FOREST VALIDATION RESULTS")
    print("=" * 50)

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------
    # 10. Feature importance
    # --------------------------------------------------

    print("\nFeature Importance:")

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print(importance_df.to_string(index=False))

    # --------------------------------------------------
    # 11. Save model and results
    # --------------------------------------------------

    print("\nSaving trained model...")

    joblib.dump(rf_model, MODEL_PATH, compress=3)

    with open(REPORT_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    elapsed_time = time.time() - start_time

    print("\nModel saved at:", MODEL_PATH)
    print("Metrics saved at:", REPORT_PATH)

    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
