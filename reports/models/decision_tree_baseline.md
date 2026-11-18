# Decision Tree Baseline Model – PathSnap

## 1. Model Overview

**Model:** Decision Tree Classifier  
**Purpose:** Predict whether a candidate road segment is the correct match for a GPS observation.

A Decision Tree is a supervised machine learning algorithm that makes predictions using a sequence of feature-based decisions. It splits the dataset into smaller groups based on feature values until it reaches a prediction.

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

The same preprocessed training and validation datasets used for the Random Forest and XGBoost baselines were used in this experiment.

## 3. Model Configuration

| Parameter | Value |
|---|---|
| criterion | gini |
| max_depth | 18 |
| min_samples_split | 10 |
| min_samples_leaf | 5 |
| class_weight | balanced |
| random_state | 42 |

## 4. Training Performance

**Training Time:** 16.48 seconds

The Decision Tree was trained using the preprocessed training dataset and evaluated on the validation dataset.

Class balancing was enabled to account for the imbalance between positive and negative target classes.

## 5. Evaluation Metrics

| Metric | Validation Result |
|---|---:|
| Accuracy | 91.65% |
| Precision | 57.29% |
| Recall | 91.71% |
| F1-Score | 70.52% |
| ROC-AUC | 95.39% |
| PR-AUC | 81.07% |
| Top-1 Candidate Accuracy | 82.01% |

## 6. Feature Importance

| Feature | Importance |
|---|---:|
| candidate_rank | 79.52% |
| road_length_m | 6.80% |
| heading_difference_deg | 6.74% |
| gps_speed_mps | 3.56% |
| distance_m | 3.38% |

**Observation:** Candidate rank is the most influential feature, contributing approximately 79.52% of the total feature importance.

Feature importance indicates how the model uses features; it does not establish a causal relationship.

## 7. Observations

1. The model achieved 91.65% validation accuracy.
2. Recall of 91.71% indicates that the model identifies a high proportion of actual positive candidates.
3. Precision of 57.29% indicates that a considerable number of predicted positive candidates are false positives.
4. ROC-AUC of 95.39% and PR-AUC of 81.07% indicate good classification performance.
5. Top-1 candidate accuracy of 82.01% shows that the correct candidate was ranked first for approximately 82.01% of labeled observations.
6. The model relies heavily on candidate_rank compared with the other four features.

## 8. Conclusion

The Decision Tree model provides a useful interpretable baseline for the PathSnap candidate classification task.

It achieved high recall but lower precision compared with Random Forest and XGBoost. Its F1-score and top-1 candidate accuracy were also lower than those of the other two models.

This suggests that a single decision tree has limitations in balancing false positives and false negatives for this dataset.

The Decision Tree is retained as the third baseline model for comparison with other machine learning algorithms.

## 9. Saved Artifacts

- Trained model: `models/artifacts/decision_tree.joblib`
- Evaluation metrics: `reports/models/decision_tree_metrics.json`
- Training script: `src/models/train_decision_tree.py`