"""Train and evaluate Logistic Regression on the shared candidate dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    "candidate_rank",
    "distance_m",
    "heading_difference_deg",
    "road_length_m",
    "gps_speed_mps",
)
IDENTIFIER_COLUMNS = ("trip_id", "observation_index", "candidate_arc_id")
EVALUATION_SPLITS = ("train", "validation", "test")
EXPECTED_TRIP_COUNTS = {
    "train": 64,
    "validation": 8,
    "test": 8,
    "student_test_dev": 10,
}


def _validate_dataset(dataset: pd.DataFrame) -> dict[str, int]:
    required_columns = {
        *IDENTIFIER_COLUMNS,
        *FEATURE_COLUMNS,
        "target",
        "dataset_split",
    }
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if dataset.empty:
        raise ValueError("The candidate dataset has no rows")
    if dataset[list(IDENTIFIER_COLUMNS)].isna().any().any():
        raise ValueError("Candidate and observation identifiers must not be missing")
    if dataset["target"].isna().any() or not dataset["target"].isin((0, 1)).all():
        raise ValueError("The target column must contain only nonmissing 0/1 labels")

    split_names = set(dataset["dataset_split"].dropna().unique())
    if split_names != set(EXPECTED_TRIP_COUNTS):
        raise ValueError(
            "Unexpected dataset splits: "
            f"expected {sorted(EXPECTED_TRIP_COUNTS)}, got {sorted(split_names)}"
        )
    if dataset["dataset_split"].isna().any():
        raise ValueError("The dataset_split column must not be missing")

    trip_split_counts = dataset.groupby("trip_id")["dataset_split"].nunique()
    if (trip_split_counts != 1).any():
        raise ValueError("Every trip must belong to exactly one dataset split")
    trip_counts = dataset.groupby("dataset_split")["trip_id"].nunique().to_dict()
    if trip_counts != EXPECTED_TRIP_COUNTS:
        raise ValueError(
            f"Unexpected trip counts by split: expected {EXPECTED_TRIP_COUNTS}, "
            f"got {trip_counts}"
        )
    return {split: int(trip_counts[split]) for split in EXPECTED_TRIP_COUNTS}


def _split_metrics(
    truth: pd.Series, predictions: pd.Series, probabilities: pd.Series
) -> dict[str, object]:
    if truth.nunique() != 2:
        raise ValueError("Each evaluated split must contain both target classes")
    matrix = confusion_matrix(truth, predictions, labels=(0, 1))
    return {
        "candidate_rows": int(len(truth)),
        "positive_rows": int(truth.sum()),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "average_precision": float(average_precision_score(truth, probabilities)),
        "confusion_matrix": {
            "labels": [0, 1],
            "rows_actual_columns_predicted": matrix.tolist(),
        },
    }


def train_and_evaluate(
    dataset_path: str | Path, output_dir: str | Path
) -> dict[str, object]:
    """Fit on train trips, evaluate agreed splits, and save model artifacts."""
    data_path = Path(dataset_path)
    output_path = Path(output_dir)
    dataset = pd.read_csv(data_path, dtype={"trip_id": "string"})
    trip_counts = _validate_dataset(dataset)

    for column in FEATURE_COLUMNS:
        dataset[column] = pd.to_numeric(dataset[column], errors="raise")

    training = dataset.loc[dataset["dataset_split"] == "train"]
    training_target = training["target"].astype(int)
    if training_target.nunique() != 2:
        raise ValueError("The train split must contain both target classes")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )
    model.fit(training.loc[:, FEATURE_COLUMNS], training_target)

    predicted_target = model.predict(dataset.loc[:, FEATURE_COLUMNS])
    positive_class_index = list(model.classes_).index(1)
    positive_probability = model.predict_proba(
        dataset.loc[:, FEATURE_COLUMNS]
    )[:, positive_class_index]
    dataset["predicted_target"] = predicted_target
    dataset["probability_target_1"] = positive_probability

    metrics_by_split = {}
    for split in EVALUATION_SPLITS:
        rows = dataset.loc[dataset["dataset_split"] == split]
        metrics_by_split[split] = _split_metrics(
            rows["target"].astype(int),
            rows["predicted_target"],
            rows["probability_target_1"],
        )

    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "logistic_regression_pipeline.joblib"
    predictions_path = output_path / "candidate_predictions.csv"
    metrics_path = output_path / "metrics.json"

    joblib.dump(model, model_path)
    predictions = dataset[
        [
            *IDENTIFIER_COLUMNS,
            "dataset_split",
            "target",
            "predicted_target",
            "probability_target_1",
        ]
    ]
    predictions.to_csv(predictions_path, index=False)

    metrics = {
        "dataset_path": str(data_path.resolve()),
        "dataset_rows": int(len(dataset)),
        "trip_counts_by_split": trip_counts,
        "fit_split": "train",
        "evaluated_splits": list(EVALUATION_SPLITS),
        "prediction_rows_by_split": {
            split: int((dataset["dataset_split"] == split).sum())
            for split in EXPECTED_TRIP_COUNTS
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "pipeline": [
            "median_imputer",
            "standard_scaler",
            "logistic_regression",
        ],
        "metrics_by_split": metrics_by_split,
        "artifacts": {
            "model": model_path.name,
            "candidate_predictions": predictions_path.name,
        },
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(project_root / "derived" / "candidate_dataset.csv"),
        help="Path to the canonical shared candidate CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "artifacts" / "logistic_regression"),
        help="Directory for the model, metrics, and candidate predictions",
    )
    args = parser.parse_args()
    metrics = train_and_evaluate(args.dataset, args.output_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
