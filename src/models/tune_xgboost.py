
import json
import time
from pathlib import Path

import optuna
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
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

BASELINE_METRICS_PATH = REPORT_DIR / "xgboost_metrics.json"

TUNED_MODEL_PATH = ARTIFACT_DIR / "xgboost_tuned.json"
BEST_CONFIG_PATH = ARTIFACT_DIR / "xgboost_best_config.json"
TUNED_METRICS_PATH = REPORT_DIR / "xgboost_tuned_metrics.json"
TRIALS_PATH = REPORT_DIR / "xgboost_tuning_trials.csv"
COMPARISON_PATH = REPORT_DIR / "xgboost_tuning_comparison.md"

N_TRIALS = 20
RANDOM_STATE = 42

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
# 2. LOAD DATA
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
print("Features:", FEATURE_COLUMNS)

# --------------------------------------------------
# 3. TOP-1 ACCURACY FUNCTION
# --------------------------------------------------

def calculate_top1_accuracy(data, probabilities):

    ranking_df = data[
        ID_COLUMNS + ["candidate_rank"]
    ].copy()

    ranking_df["target"] = data[TARGET_COLUMN].to_numpy()
    ranking_df["predicted_probability"] = probabilities

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

    return float(top_candidates["target"].mean())


# --------------------------------------------------
# 4. BAYESIAN OPTIMIZATION OBJECTIVE
# --------------------------------------------------

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 150, 400
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.2, log=True
        ),

        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
        ),

        "gamma": trial.suggest_float(
            "gamma", 0.0, 5.0
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 10.0, log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 20.0, log=True
        ),

        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 0.8, 3.0
        ),
    }

    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    probabilities = model.predict_proba(X_val)[:, 1]

    top1_accuracy = calculate_top1_accuracy(
        val_df,
        probabilities,
    )

    # Store additional metrics for analysis.
    predictions = (probabilities >= 0.5).astype(int)

    trial.set_user_attr(
        "f1_score",
        float(f1_score(y_val, predictions, zero_division=0)),
    )

    trial.set_user_attr(
        "pr_auc",
        float(average_precision_score(y_val, probabilities)),
    )

    trial.set_user_attr(
        "roc_auc",
        float(roc_auc_score(y_val, probabilities)),
    )

    print(
        f"Trial {trial.number + 1}/{N_TRIALS} | "
        f"Top-1 Accuracy: {top1_accuracy:.4f}"
    )

    return top1_accuracy


# --------------------------------------------------
# 5. RUN BAYESIAN OPTIMIZATION
# --------------------------------------------------

print("\nStarting Bayesian Optimization...")
print("Optimization method: Optuna TPE Sampler")
print("Number of trials:", N_TRIALS)
print("Optimization metric: Top-1 Candidate Accuracy")

optuna.logging.set_verbosity(optuna.logging.WARNING)

sampler = optuna.samplers.TPESampler(
    seed=RANDOM_STATE
)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
)

optimization_start = time.time()

study.optimize(
    objective,
    n_trials=N_TRIALS,
    gc_after_trial=True,
)

optimization_time = time.time() - optimization_start

print("\nBayesian Optimization completed!")
print(f"Total optimization time: {optimization_time:.2f} seconds")

# --------------------------------------------------
# 6. SAVE TRIAL RESULTS
# --------------------------------------------------

trials_df = study.trials_dataframe()

trials_df.to_csv(
    TRIALS_PATH,
    index=False,
)

print("Trial results saved to:", TRIALS_PATH)

# --------------------------------------------------
# 7. GET BEST HYPERPARAMETERS
# --------------------------------------------------

best_params = study.best_params

print("\nBest hyperparameters:")

for name, value in best_params.items():
    print(f"{name}: {value}")

print(
    f"\nBest validation Top-1 Accuracy: "
    f"{study.best_value:.4f}"
)

best_config = {
    "model": "Tuned XGBoost",
    "optimization_method": "Optuna TPE Bayesian Optimization",
    "optimization_metric": "top1_candidate_accuracy",
    "number_of_trials": N_TRIALS,
    "best_top1_candidate_accuracy": float(study.best_value),
    "optimization_time_seconds": round(optimization_time, 2),
    "features": FEATURE_COLUMNS,
    "hyperparameters": best_params,
    "fixed_parameters": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 4,
        "random_state": RANDOM_STATE,
    },
}

with open(BEST_CONFIG_PATH, "w") as file:
    json.dump(best_config, file, indent=4)

print("Best configuration saved to:", BEST_CONFIG_PATH)

# --------------------------------------------------
# 8. TRAIN FINAL TUNED MODEL
# --------------------------------------------------

print("\nTraining final tuned XGBoost model...")

final_model = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=4,
    random_state=RANDOM_STATE,
)

final_training_start = time.time()

final_model.fit(
    X_train,
    y_train,
    verbose=False,
)

final_training_time = time.time() - final_training_start

print(
    f"Final model training time: "
    f"{final_training_time:.2f} seconds"
)

# --------------------------------------------------
# 9. EVALUATE FINAL TUNED MODEL
# --------------------------------------------------

print("\nEvaluating tuned model...")

y_prob = final_model.predict_proba(X_val)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_val, y_prob)
pr_auc = average_precision_score(y_val, y_prob)

top1_accuracy = calculate_top1_accuracy(
    val_df,
    y_prob,
)

feature_importance = {
    feature: float(importance)
    for feature, importance in zip(
        FEATURE_COLUMNS,
        final_model.feature_importances_,
    )
}

# --------------------------------------------------
# 10. SAVE TUNED MODEL
# --------------------------------------------------

final_model.save_model(str(TUNED_MODEL_PATH))

print("\nTuned model saved to:", TUNED_MODEL_PATH)

# --------------------------------------------------
# 11. SAVE TUNED METRICS
# --------------------------------------------------

tuned_metrics = {
    "model": "Tuned XGBoost",
    "optimization_method": "Optuna TPE Bayesian Optimization",
    "optimization_trials": N_TRIALS,
    "optimization_time_seconds": round(optimization_time, 2),
    "training_time_seconds": round(final_training_time, 2),
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
    "configuration": best_params,
}

with open(TUNED_METRICS_PATH, "w") as file:
    json.dump(tuned_metrics, file, indent=4)

print("Tuned metrics saved to:", TUNED_METRICS_PATH)

# --------------------------------------------------
# 12. BASELINE VS TUNED COMPARISON
# --------------------------------------------------

with open(BASELINE_METRICS_PATH, "r") as file:
    baseline_metrics = json.load(file)

comparison_metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "top1_candidate_accuracy",
]

comparison_lines = [
    "# XGBoost: Baseline vs Bayesian Optimized",
    "",
    "| Metric | Baseline XGBoost | Tuned XGBoost |",
    "|---|---:|---:|",
]

for metric in comparison_metrics:

    baseline_value = baseline_metrics[metric]
    tuned_value = tuned_metrics[metric]

    comparison_lines.append(
        f"| {metric} | {baseline_value:.4f} | {tuned_value:.4f} |"
    )

comparison_lines.extend([
    "",
    "## Best Optimization Configuration",
    "",
    "```json",
    json.dumps(best_params, indent=4),
    "```",
])

with open(COMPARISON_PATH, "w") as file:
    file.write("\n".join(comparison_lines))

print("Comparison report saved to:", COMPARISON_PATH)

# --------------------------------------------------
# 13. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\n========== TUNED XGBOOST RESULTS ==========")

print(f"Accuracy:             {accuracy:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"Recall:               {recall:.4f}")
print(f"F1 Score:             {f1:.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")
print(f"PR-AUC:               {pr_auc:.4f}")
print(f"Top-1 Candidate Acc:  {top1_accuracy:.4f}")

print("\n========== BASELINE VS TUNED ==========")

print(
    f"Baseline Top-1: {baseline_metrics['top1_candidate_accuracy']:.4f}"
)

print(
    f"Tuned Top-1:    {top1_accuracy:.4f}"
)

print("\nBayesian optimization pipeline completed successfully!")
