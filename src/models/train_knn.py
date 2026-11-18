
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports" / "models"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "knn_model.joblib"
METRICS_PATH = REPORT_DIR / "knn_metrics.json"

# --------------------------------------------------
# 2. FEATURES AND TARGET
# --------------------------------------------------

FEATURE_COLUMNS = [
    "candidate_rank",
    "distance_m",
    "heading_difference_deg",
    "road_length_m",
    "gps_speed_mps",
]

TARGET_COLUMN = "target"

ID_COLUMNS = [
    "trip_id",
    "observation_index",
]

# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

print("Loading preprocessed datasets...")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df = pd.read_parquet(DATA_DIR / "validation.parquet")

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_val = val_df[FEATURE_COLUMNS]
y_val = val_df[TARGET_COLUMN]

print("Full training rows:", len(train_df))
print("Validation rows:", len(val_df))

# --------------------------------------------------
# 4. CREATE STRATIFIED TRAINING SAMPLE
# --------------------------------------------------

SAMPLE_SIZE = 100_000

print(f"\nSelecting {SAMPLE_SIZE:,} stratified training rows...")

X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train,
    y_train,
    train_size=SAMPLE_SIZE,
    stratify=y_train,
    random_state=42,
)

print("KNN training sample:", len(X_train_sample))
print("Positive samples:", int(y_train_sample.sum()))
print("Negative samples:", int((y_train_sample == 0).sum()))

# --------------------------------------------------
# 5. INITIALIZE KNN PIPELINE
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            algorithm="kd_tree",
            leaf_size=40,
            p=2,
            n_jobs=4,
        ),
    ),
])

print("\nKNN configuration:")
print(model)

# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nTraining KNN model...")

start_time = time.time()

model.fit(X_train_sample, y_train_sample)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")

# --------------------------------------------------
# 7. GENERATE VALIDATION PREDICTIONS IN BATCHES
# --------------------------------------------------

print("\nGenerating validation predictions in batches...")

BATCH_SIZE = 5_000
probability_batches = []

prediction_start = time.time()

for start in range(0, len(X_val), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(X_val))

    batch = X_val.iloc[start:end]

    probabilities = model.predict_proba(batch)[:, 1]

    probability_batches.extend(probabilities)

    print(f"Processed {end:,} / {len(X_val):,} rows")

y_prob = pd.Series(probability_batches).to_numpy()
y_pred = (y_prob >= 0.5).astype(int)

prediction_time = time.time() - prediction_start

print(f"Prediction time: {prediction_time:.2f} seconds")

# --------------------------------------------------
# 8. CLASSIFICATION METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_val, y_prob)
pr_auc = average_precision_score(y_val, y_prob)

print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))

# --------------------------------------------------
# 9. TOP-1 CANDIDATE ACCURACY
# --------------------------------------------------

ranking_df = val_df[
    ID_COLUMNS + ["candidate_rank"]
].copy()

ranking_df["target"] = y_val.to_numpy()
ranking_df["predicted_probability"] = y_prob

ranking_df = ranking_df.sort_values(
    by=[
        "trip_id",
        "observation_index",
        "predicted_probability",
        "candidate_rank",
    ],
    ascending=[True, True, False, True],
)

top_candidates = ranking_df.drop_duplicates(
    subset=ID_COLUMNS,
    keep="first",
)

top1_accuracy = top_candidates["target"].mean()

# --------------------------------------------------
# 10. SAVE MODEL
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")

# --------------------------------------------------
# 11. SAVE METRICS
# --------------------------------------------------

metrics = {
    "model": "K-Nearest Neighbors",
    "training_rows": len(X_train_sample),
    "full_training_rows": len(train_df),
    "validation_rows": len(val_df),
    "training_time_seconds": round(training_time, 2),
    "prediction_time_seconds": round(prediction_time, 2),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "top1_candidate_accuracy": float(top1_accuracy),
    "configuration": {
        "n_neighbors": 5,
        "weights": "distance",
        "algorithm": "kd_tree",
        "leaf_size": 40,
        "p": 2,
        "scaling": "StandardScaler",
        "sample_size": SAMPLE_SIZE,
        "random_state": 42,
    },
    "feature_importance": None,
}

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)

print(f"Metrics saved to: {METRICS_PATH}")

# --------------------------------------------------
# 12. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== KNN RESULTS ==========")

print(f"Accuracy:             {accuracy:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"Recall:               {recall:.4f}")
print(f"F1 Score:             {f1:.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")
print(f"PR-AUC:               {pr_auc:.4f}")
print(f"Top-1 Candidate Acc:  {top1_accuracy:.4f}")
print(f"Training Time:        {training_time:.2f} seconds")
print(f"Prediction Time:      {prediction_time:.2f} seconds")

print("\nKNN training pipeline completed successfully!")
