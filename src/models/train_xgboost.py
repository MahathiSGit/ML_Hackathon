
import json
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

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

MODEL_PATH = ARTIFACT_DIR / "xgboost_model.json"
METRICS_PATH = REPORT_DIR / "xgboost_metrics.json"

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
# 3. LOAD PREPROCESSED DATA
# --------------------------------------------------

print("Loading preprocessed datasets...")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df = pd.read_parquet(DATA_DIR / "validation.parquet")

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_val = val_df[FEATURE_COLUMNS]
y_val = val_df[TARGET_COLUMN]

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Number of features:", len(FEATURE_COLUMNS))

# --------------------------------------------------
# 4. INITIALIZE XGBOOST MODEL
# --------------------------------------------------

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=4,
    random_state=42,
    scale_pos_weight=1,
)

print("\nXGBoost model configuration:")
print(model)

# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

print("\nTraining XGBoost model...")

start_time = time.time()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=25,
)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")

# --------------------------------------------------
# 6. PREDICTIONS
# --------------------------------------------------

print("\nGenerating validation predictions...")

y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# --------------------------------------------------
# 7. CLASSIFICATION METRICS
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
# 8. TOP-1 CANDIDATE ACCURACY
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
# 9. FEATURE IMPORTANCE
# --------------------------------------------------

feature_importance = {
    feature: float(importance)
    for feature, importance in zip(
        FEATURE_COLUMNS,
        model.feature_importances_,
    )
}

feature_importance = dict(
    sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )
)

# --------------------------------------------------
# 10. SAVE MODEL
# --------------------------------------------------

model.save_model(str(MODEL_PATH))

print(f"\nModel saved to: {MODEL_PATH}")

# --------------------------------------------------
# 11. SAVE METRICS
# --------------------------------------------------

metrics = {
    "model": "XGBoost",
    "training_time_seconds": round(training_time, 2),
    "training_rows": len(train_df),
    "validation_rows": len(val_df),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "top1_candidate_accuracy": float(top1_accuracy),
    "feature_importance": feature_importance,
    "configuration": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": 4,
        "scale_pos_weight": 1,
        "random_state": 42,
    },
}

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)

print(f"Metrics saved to: {METRICS_PATH}")

# --------------------------------------------------
# 12. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== XGBOOST RESULTS ==========")

print(f"Accuracy:             {accuracy:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"Recall:               {recall:.4f}")
print(f"F1 Score:             {f1:.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")
print(f"PR-AUC:               {pr_auc:.4f}")
print(f"Top-1 Candidate Acc:  {top1_accuracy:.4f}")

print("\nFeature Importance:")

for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")

print("\nXGBoost training pipeline completed successfully!")
