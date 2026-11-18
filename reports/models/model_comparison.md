# PathSnap - Model Comparison Report

## 1. Objective

Compare the baseline ML models using their validation performance to select a suitable road-candidate classification and ranking model.

## 2. Validation Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | Top-1 Accuracy | Training Time (s) | Training Rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 86.20% | 43.88% | 95.75% | 60.18% | 97.22% | 82.18% | 86.73% | 1.13 | 1536611 |
| Tuned XGBoost | 95.63% | 76.70% | 86.01% | 81.09% | 98.09% | 87.47% | 86.72% | 11.08 | 1536611 |
| XGBoost | 96.00% | 83.66% | 78.62% | 81.06% | 98.08% | 87.53% | 86.28% | 10.75 | 1536611 |
| Random Forest | 94.03% | 66.47% | 91.16% | 76.88% | 98.00% | 87.21% | 85.52% | 16.02 | 1536611 |
| Decision Tree | 91.65% | 57.29% | 91.71% | 70.52% | 95.39% | 81.07% | 82.01% | 16.48 | 1536611 |
| K-Nearest Neighbors | 94.61% | 77.76% | 70.75% | 74.09% | 93.54% | 76.15% | 80.09% | 0.08 | 100000 |

## 3. Best Model by Evaluation Metric

| Metric | Best Model | Score |
| --- | --- | ---: |
| Accuracy | XGBoost | 96.00% |
| Precision | XGBoost | 83.66% |
| Recall | Logistic Regression | 95.75% |
| F1 Score | Tuned XGBoost | 81.09% |
| Roc Auc | Tuned XGBoost | 98.09% |
| Pr Auc | XGBoost | 87.53% |
| Top1 Candidate Accuracy | Logistic Regression | 86.73% |
| Training Time Seconds | K-Nearest Neighbors | 0.08 seconds |

## 4. Important Observations

- Top-1 candidate accuracy measures whether the highest-ranked candidate is the correct candidate for each observation.
- Precision and recall measure binary classification performance at the selected classification threshold.
- PR-AUC is important because the dataset contains considerably more negative than positive candidates.
- Training time should be interpreted alongside the amount of training data used.
- KNN was trained on a 100,000-row stratified sample, whereas the other baseline models used the full training dataset. Its training time and performance are therefore not a fully controlled comparison.
- These results are from the validation dataset, not the final test dataset.

## 5. Model Selection Considerations

For PathSnap, Top-1 candidate accuracy is especially relevant because the objective is to select the most likely road candidate for each GPS observation.

However, final selection should also consider F1-score, PR-AUC, prediction time, and performance on unseen test data.

## 6. Next Steps

1. Complete the remaining baseline models, including LightGBM if selected.
2. Compare the models using validation results.
3. Tune the most promising models.
4. Select one final model.
5. Evaluate the selected model on the untouched test dataset.
