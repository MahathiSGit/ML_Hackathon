# XGBoost Baseline Model – PathSnap

## 1. Model Overview

**Model:** XGBoost Classifier  
**Purpose:** Predict whether a candidate road segment is the correct match for a GPS observation.

XGBoost is a gradient-boosting algorithm that builds decision trees sequentially, with each new tree improving upon the errors made by previous trees.

## 2. Dataset Details

| Dataset | Number of Rows |
|---|---:|
| Training | 1,536,611 |
| Validation | 157,591 |
| Features | 5 |

**Input Features:**
- candidate_rank
- distance_m
- heading_difference_deg
- road_length_m
- gps_speed_mps

**Target:** `target` (0 = Negative, 1 = Positive)

The same preprocessed training and validation datasets used for the Random Forest baseline were used for this experiment.

## 3. Model Configuration

| Parameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | binary:logistic |
| eval_metric | logloss |
| tree_method | hist |
| n_jobs | 4 |
| scale_pos_weight | 1 |
| random_state | 42 |

## 4. Training Performance

**Training Time:** 10.75 seconds

The validation log loss decreased from 0.30060 at the beginning of training to approximately 0.10591 around iteration 125. It remained relatively stable towards the end.

## 5. Evaluation Metrics

| Metric | Validation Result |
|---|---:|
| Accuracy | 96.00% |
| Precision | 83.66% |
| Recall | 78.62% |
| F1-Score | 81.06% |
| ROC-AUC | 98.08% |
| PR-AUC | 87.53% |
| Top-1 Candidate Accuracy | 86.28% |

## 6. Feature Importance

| Feature | Importance |
|---|---:|
| candidate_rank | 62.67% |
| heading_difference_deg | 15.35% |
| distance_m | 14.74% |
| gps_speed_mps | 3.82% |
| road_length_m | 3.42% |

**Observation:** Candidate rank is the most influential feature in the XGBoost model, followed by heading difference and distance.

Feature importance indicates how the model uses features; it does not establish a causal relationship.

## 7. Observations

1. The model achieved 96% validation accuracy.
2. Precision of 83.66% indicates that most candidates predicted as positive were actually positive.
3. Recall of 78.62% indicates that some actual positive candidates were missed.
4. ROC-AUC of 98.08% and PR-AUC of 87.53% indicate strong classification performance.
5. Top-1 candidate accuracy of 86.28% shows that the correct candidate was ranked first for approximately 86.28% of labeled observations.
6. XGBoost completed training faster than the Random Forest baseline in this experiment.

## 8. Conclusion

The XGBoost model achieved strong validation performance and outperformed the Random Forest baseline in accuracy, precision, F1-score, ROC-AUC, PR-AUC, and top-1 candidate accuracy.

However, Random Forest achieved higher recall. Therefore, both models have different strengths, and further evaluation is needed before selecting the final model.

XGBoost is retained as the second baseline model for PathSnap.

## 9. Saved Artifacts

- Trained model: `models/artifacts/xgboost_model.json`
- Evaluation metrics: `reports/models/xgboost_metrics.json`
- Training script: `src/models/train_xgboost.py`