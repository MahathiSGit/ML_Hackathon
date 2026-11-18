"""Train and evaluate KNN on Mahima's shared candidate-level dataset.

This module consumes the shared candidate CSV as-is. It does not build or
relabel candidates. Identifiers and split metadata are retained only for
prediction outputs and are never used as model features.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "candidate_rank",
    "distance_m",
    "heading_difference_deg",
    "road_length_m",
    "gps_speed_mps",
]
IDENTIFIER_COLUMNS = ["trip_id", "observation_index", "candidate_arc_id"]
K_VALUES = (3, 5, 9)


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    trip_id: np.ndarray
    observation_index: np.ndarray
    candidate_arc_id: np.ndarray
    candidate_rank: np.ndarray


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def load_split(dataset_path: str | Path, split: str, expected_rows: int | None = None) -> SplitData:
    """Read one existing split from the shared CSV; preserve its original IDs."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Shared candidate dataset not found: {path}")

    if expected_rows is None:
        with _open_csv(path) as stream:
            expected_rows = sum(
                1 for row in csv.DictReader(stream) if row["dataset_split"] == split
            )

    X = np.empty((expected_rows, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.empty(expected_rows, dtype=np.uint8)
    trip_id = np.empty(expected_rows, dtype="U32")
    observation_index = np.empty(expected_rows, dtype=np.int64)
    candidate_arc_id = np.empty(expected_rows, dtype=np.int64)
    candidate_rank = np.empty(expected_rows, dtype=np.int32)

    row_index = 0
    with _open_csv(path) as stream:
        for row in csv.DictReader(stream):
            if row["dataset_split"] != split:
                continue
            if row_index >= expected_rows:
                raise ValueError(f"Manifest row count for {split} is too small")
            for feature_index, column in enumerate(FEATURE_COLUMNS):
                value = row[column]
                X[row_index, feature_index] = np.nan if value == "" else float(value)
            y[row_index] = int(row["target"])
            trip_id[row_index] = row["trip_id"]
            observation_index[row_index] = int(row["observation_index"])
            candidate_arc_id[row_index] = int(row["candidate_arc_id"])
            candidate_rank[row_index] = int(row["candidate_rank"])
            row_index += 1
    if row_index != expected_rows:
        raise ValueError(
            f"Manifest expected {expected_rows} {split} rows, found {row_index}"
        )
    if row_index == 0:
        raise ValueError(f"No rows found for split {split}")
    return SplitData(X, y, trip_id, observation_index, candidate_arc_id, candidate_rank)


def make_pipeline(k: int) -> Pipeline:
    """Impute missing shared features, scale them, then run KNN."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=k,
            algorithm="kd_tree",
            metric="euclidean",
            n_jobs=-1,
        )),
    ])


def _positive_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    classes = model.named_steps["knn"].classes_
    positive_column = int(np.flatnonzero(classes == 1)[0])
    return model.predict_proba(X)[:, positive_column]


def _top1_accuracy(data: SplitData, scores: np.ndarray) -> float:
    """Choose one candidate per observation by score, then rank, then arc ID."""
    order = np.lexsort((
        data.candidate_arc_id,
        data.candidate_rank,
        -scores,
        data.observation_index,
        data.trip_id,
    ))
    sorted_trip = data.trip_id[order]
    sorted_obs = data.observation_index[order]
    starts = np.r_[
        0,
        np.flatnonzero(
            (sorted_trip[1:] != sorted_trip[:-1])
            | (sorted_obs[1:] != sorted_obs[:-1])
        ) + 1,
    ]
    top_rows = order[starts]
    return float(np.mean(data.y[top_rows]))


def evaluate(data: SplitData, scores: np.ndarray) -> dict:
    predictions = (scores > 0.5).astype(np.uint8)
    metrics = {
        "candidate_accuracy_at_0_5": float(accuracy_score(data.y, predictions)),
        "candidate_precision_at_0_5": float(precision_score(
            data.y, predictions, zero_division=0
        )),
        "candidate_recall_at_0_5": float(recall_score(
            data.y, predictions, zero_division=0
        )),
        "candidate_f1_at_0_5": float(f1_score(
            data.y, predictions, zero_division=0
        )),
        "top1_accuracy_per_observation": _top1_accuracy(data, scores),
        "positive_candidate_rate": float(np.mean(data.y)),
        "candidate_rows": int(len(data.y)),
    }
    if np.unique(data.y).size == 2:
        metrics["candidate_roc_auc"] = float(roc_auc_score(data.y, scores))
        metrics["candidate_average_precision"] = float(
            average_precision_score(data.y, scores)
        )
    return metrics


def save_predictions(path: Path, data: SplitData, scores: np.ndarray, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "trip_id",
            "observation_index",
            "candidate_arc_id",
            "candidate_rank",
            "target",
            "predicted_target",
            "positive_probability",
            "dataset_split",
        ])
        for i, score in enumerate(scores):
            writer.writerow([
                data.trip_id[i],
                int(data.observation_index[i]),
                int(data.candidate_arc_id[i]),
                int(data.candidate_rank[i]),
                int(data.y[i]),
                int(score > 0.5),
                float(score),
                split,
            ])


def train_and_evaluate(
    dataset_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    k_values: Iterable[int] = K_VALUES,
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    counts = manifest["counts_by_split"]
    train = load_split(dataset_path, "train", counts["train"]["candidate_rows"])
    validation = load_split(
        dataset_path, "validation", counts["validation"]["candidate_rows"]
    )

    selection_results = {}
    selected_model = None
    selected_scores = None
    selected_k = None
    best_key = None

    for k in k_values:
        model = make_pipeline(int(k))
        model.fit(train.X, train.y)
        scores = _positive_scores(model, validation.X)
        metrics = evaluate(validation, scores)
        selection_results[str(k)] = metrics
        selection_key = (
            metrics["top1_accuracy_per_observation"],
            metrics.get("candidate_average_precision", 0.0),
            metrics["candidate_f1_at_0_5"],
            -int(k),
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            selected_model = model
            selected_scores = scores
            selected_k = int(k)

    if selected_model is None:
        raise ValueError("At least one positive K value is required")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_predictions(
        output / "validation_predictions.csv.gz",
        validation,
        selected_scores,
        "validation",
    )
    validation_metrics = evaluate(validation, selected_scores)

    # Final KNN fit uses train + validation only after choosing K. Test and
    # student_test_dev remain excluded from both model and preprocessing fits.
    combined_X = np.concatenate((train.X, validation.X), axis=0)
    combined_y = np.concatenate((train.y, validation.y), axis=0)
    final_model = make_pipeline(selected_k)
    final_model.fit(combined_X, combined_y)

    evaluation_metrics = {}
    prediction_counts = {}
    for split in ("test", "student_test_dev"):
        data = load_split(dataset_path, split, counts[split]["candidate_rows"])
        scores = _positive_scores(final_model, data.X)
        evaluation_metrics[split] = evaluate(data, scores)
        prediction_counts[split] = len(scores)
        save_predictions(
            output / f"{split}_predictions.csv.gz", data, scores, split
        )

    model_path = output / "knn_model.joblib"
    joblib.dump({
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "identifier_columns": IDENTIFIER_COLUMNS,
        "target_column": manifest["target_column"],
        "target_definition": manifest["target_definition"],
        "selected_k": selected_k,
        "fit_splits": ["train", "validation"],
    }, model_path, compress=3)

    results = {
        "model": "KNeighborsClassifier",
        "feature_columns": FEATURE_COLUMNS,
        "identifier_columns": IDENTIFIER_COLUMNS,
        "target_column": manifest["target_column"],
        "target_definition": manifest["target_definition"],
        "k_values_tried": [int(k) for k in k_values],
        "selection_metric": "validation top1_accuracy_per_observation",
        "selected_k": selected_k,
        "validation_metrics": validation_metrics,
        "validation_by_k": selection_results,
        "evaluation_metrics": evaluation_metrics,
        "fit_splits_for_saved_model": ["train", "validation"],
        "artifacts": {
            "model": str(model_path),
            "validation_predictions": str(output / "validation_predictions.csv.gz"),
            "test_predictions": str(output / "test_predictions.csv.gz"),
            "student_test_dev_predictions": str(
                output / "student_test_dev_predictions.csv.gz"
            ),
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/raw/candidate_dataset.csv.gz",
        help="Path to Mahima's shared candidate CSV (plain or gzip)",
    )
    parser.add_argument(
        "--manifest",
        default="data/raw/candidate_dataset_manifest.json",
        help="Path to the shared dataset manifest",
    )
    parser.add_argument("--output-dir", default="artifacts/knn")
    args = parser.parse_args()
    results = train_and_evaluate(args.dataset, args.manifest, args.output_dir)
    print(json.dumps({
        "selected_k": results["selected_k"],
        "validation_metrics": results["validation_metrics"],
        "evaluation_metrics": results["evaluation_metrics"],
        "artifacts": results["artifacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
