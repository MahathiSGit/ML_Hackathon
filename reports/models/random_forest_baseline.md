# PathSnap – Random Forest Baseline Experiment

## 1. Model Configuration

| Parameter | Value |
|---|---:|
| Algorithm | Random Forest Classifier |
| Number of trees | 50 |
| Maximum depth | 18 |
| Minimum samples split | 10 |
| Minimum samples leaf | 5 |
| Maximum features | sqrt |
| Class weighting | balanced_subsample |
| Random state | 42 |

## 2. Dataset Information

| Parameter | Value |
|---|---:|
| Training samples | 1,536,611 |
| Validation samples | 157,591 |
| Number of features | 5 |
| Training time | 16.02 seconds |

### Features Used
1. candidate_rank
2. distance_m
3. heading_difference_deg
4. road_length_m
5. gps_speed_mps

## 3. Validation Performance

| Metric | Score |
|---|---:|
| Accuracy | 94.03% |
| Precision | 66.47% |
| Recall | 91.16% |
| F1-score | 76.88% |
| ROC-AUC | 98.00% |
| PR-AUC | 87.21% |
| Top-1 Candidate Accuracy | 85.52% |

## 4. Feature Importance

| Feature | Importance |
|---|---:|
| candidate_rank | 45.00% |
| distance_m | 30.53% |
| road_length_m | 10.01% |
| heading_difference_deg | 9.94% |
| gps_speed_mps | 4.52% |

## 5. Observations

- The model achieved 94.03% validation accuracy.
- Recall of 91.16% indicates that most correct candidates were identified.
- ROC-AUC of 98% indicates strong classification discrimination.
- Top-1 candidate accuracy was 85.52%.
- candidate_rank and distance_m were the two most influential features according to impurity-based feature importance.
- Precision of 66.47% indicates scope for improving the classification threshold and precision-recall balance.

## 6. Conclusion

The Random Forest model provides a promising baseline for the PathSnap road candidate selection task. Further experiments with other ML algorithms, hyperparameter tuning and feature analysis will be conducted to identify the best-performing model.

**Experiment status:** Baseline completed.  
**Evaluation dataset:** Validation split.
