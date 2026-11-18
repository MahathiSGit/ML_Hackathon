
import json
import csv
from pathlib import Path

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "reports" / "models"

CSV_OUTPUT = METRICS_DIR / "model_comparison.csv"
MD_OUTPUT = METRICS_DIR / "model_comparison.md"

# Metrics where higher values are better
HIGHER_IS_BETTER = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "top1_candidate_accuracy",
]

# Training time: lower is better
LOWER_IS_BETTER = ["training_time_seconds"]

# --------------------------------------------------
# 2. Load all model metrics
# --------------------------------------------------

model_results = []

for file_path in sorted(METRICS_DIR.glob("*_metrics.json")):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        if "model" not in metrics:
            print(f"Skipping {file_path.name}: model name missing")
            continue

        model_results.append(metrics)
        print(f"Loaded: {metrics['model']}")

    except (json.JSONDecodeError, OSError) as error:
        print(f"Could not read {file_path.name}: {error}")

if not model_results:
    raise RuntimeError("No valid model metric files found.")

# --------------------------------------------------
# 3. Create comparison table
# --------------------------------------------------

comparison_columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "top1_candidate_accuracy",
    "training_time_seconds",
    "training_rows",
]

comparison_data = []

for result in model_results:
    row = {
        column: result.get(column)
        for column in comparison_columns
    }
    comparison_data.append(row)

# --------------------------------------------------
# 4. Export CSV
# --------------------------------------------------

with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=comparison_columns
    )
    writer.writeheader()
    writer.writerows(comparison_data)

print(f"\nCSV saved: {CSV_OUTPUT}")

# --------------------------------------------------
# 5. Identify best model for each metric
# --------------------------------------------------

best_models = {}

for metric in HIGHER_IS_BETTER:
    valid_results = [
        result for result in model_results
        if result.get(metric) is not None
    ]

    if valid_results:
        best = max(valid_results, key=lambda x: x[metric])
        best_models[metric] = best

for metric in LOWER_IS_BETTER:
    valid_results = [
        result for result in model_results
        if result.get(metric) is not None
    ]

    if valid_results:
        best = min(valid_results, key=lambda x: x[metric])
        best_models[metric] = best

# --------------------------------------------------
# 6. Generate Markdown comparison report
# --------------------------------------------------

headers = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "PR-AUC",
    "Top-1 Accuracy",
    "Training Time (s)",
    "Training Rows",
]

lines = [
    "# PathSnap - Model Comparison Report",
    "",
    "## 1. Objective",
    "",
    "Compare the baseline ML models using their validation performance "
    "to select a suitable road-candidate classification and ranking model.",
    "",
    "## 2. Validation Results",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]

for result in sorted(
    model_results,
    key=lambda x: x.get("top1_candidate_accuracy", 0),
    reverse=True
):
    values = [
        str(result.get("model", "Unknown")),
        f"{result['accuracy']:.2%}" if result.get("accuracy") is not None else "N/A",
        f"{result['precision']:.2%}" if result.get("precision") is not None else "N/A",
        f"{result['recall']:.2%}" if result.get("recall") is not None else "N/A",
        f"{result['f1_score']:.2%}" if result.get("f1_score") is not None else "N/A",
        f"{result['roc_auc']:.2%}" if result.get("roc_auc") is not None else "N/A",
        f"{result['pr_auc']:.2%}" if result.get("pr_auc") is not None else "N/A",
        f"{result['top1_candidate_accuracy']:.2%}" if result.get("top1_candidate_accuracy") is not None else "N/A",
        f"{result['training_time_seconds']:.2f}" if result.get("training_time_seconds") is not None else "N/A",
        str(result.get("training_rows", "N/A")),
    ]

    lines.append("| " + " | ".join(values) + " |")

# --------------------------------------------------
# 7. Best model for each metric
# --------------------------------------------------

lines.extend([
    "",
    "## 3. Best Model by Evaluation Metric",
    "",
    "| Metric | Best Model | Score |",
    "| --- | --- | ---: |",
])

for metric, result in best_models.items():
    score = result[metric]

    if metric == "training_time_seconds":
        formatted_score = f"{score:.2f} seconds"
    else:
        formatted_score = f"{score:.2%}"

    display_metric = metric.replace("_", " ").title()

    lines.append(
        f"| {display_metric} | {result['model']} | {formatted_score} |"
    )

# --------------------------------------------------
# 8. Important comparison notes
# --------------------------------------------------

lines.extend([
    "",
    "## 4. Important Observations",
    "",
    "- Top-1 candidate accuracy measures whether the highest-ranked "
    "candidate is the correct candidate for each observation.",
    "- Precision and recall measure binary classification performance "
    "at the selected classification threshold.",
    "- PR-AUC is important because the dataset contains considerably "
    "more negative than positive candidates.",
    "- Training time should be interpreted alongside the amount of "
    "training data used.",
    "- KNN was trained on a 100,000-row stratified sample, whereas "
    "the other baseline models used the full training dataset. "
    "Its training time and performance are therefore not a fully "
    "controlled comparison.",
    "- These results are from the validation dataset, not the final test dataset.",
    "",
    "## 5. Model Selection Considerations",
    "",
    "For PathSnap, Top-1 candidate accuracy is especially relevant "
    "because the objective is to select the most likely road candidate "
    "for each GPS observation.",
    "",
    "However, final selection should also consider F1-score, PR-AUC, "
    "prediction time, and performance on unseen test data.",
    "",
    "## 6. Next Steps",
    "",
    "1. Complete the remaining baseline models, including LightGBM if selected.",
    "2. Compare the models using validation results.",
    "3. Tune the most promising models.",
    "4. Select one final model.",
    "5. Evaluate the selected model on the untouched test dataset.",
    "",
])

with open(MD_OUTPUT, "w", encoding="utf-8") as file:
    file.write("\n".join(lines))

print(f"Markdown report saved: {MD_OUTPUT}")

# --------------------------------------------------
# 9. Display best model summary
# --------------------------------------------------

print("\n========== BEST MODEL BY METRIC ==========")

for metric, result in best_models.items():
    print(
        f"{metric}: {result['model']} "
        f"({result[metric]:.4f})"
    )

print("\nModel comparison completed successfully!")
