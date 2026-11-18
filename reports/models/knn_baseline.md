# K-Nearest Neighbors (KNN) Baseline Model – PathSnap

## 1. Model Overview

**Model:** K-Nearest Neighbors (KNN) Classifier  
**Purpose:** Predict whether a candidate road segment is the correct match for a GPS observation.

KNN is a supervised machine learning algorithm that classifies a data point based on the classes of its nearest neighboring data points.

It uses distance calculations to identify similar observations.

## 2. Dataset Details

| Dataset | Number of Rows |
|---|---:|
| Full Training Dataset | 1,536,611 |
| KNN Training Sample | 100,000 |
| Validation | 157,591 |
| Features | 5 |

**Input Features:**
- candidate_rank
- distance_m
- heading_difference_deg
- road_length_m
- gps_speed_mps

**Target:** `target` (0 = Negative, 1 = Positive)

A stratified sample of 100,000 rows was selected from the training dataset to reduce KNN computational requirements while preserving the original class distribution.

The full validation dataset was used for evaluation.

## 3. Model Configuration

| Parameter | Value |
|---|---|
| n_neighbors | 5 |
| weights | distance |
| algorithm | kd_tree |
| leaf_size | 40 |
| p | 2 |
| n_jobs | 4 |
| Scaling | StandardScaler |
| random_state | 42 (sampling) |

StandardScaler was used because KNN relies on distance calculations. The scaler was fitted only on the selected training sample.

## 4. Training and Prediction Performance

| Operation | Time |
|---|---:|
| Model Training | 0.08 seconds |
| Validation Prediction | 1.32 seconds |

KNN has a very short fitting time because it stores the training samples rather than learning a set of decision rules.

Prediction requires finding the nearest neighbors for each validation observation.

## 5. Evaluation Metrics

| Metric | Validation Result |
|---|---:|
| Accuracy | 94.61% |
| Precision | 77.76% |
| Recall | 70.75% |
| F1-Score | 74.09% |
| ROC-AUC | 93.54% |
| PR-AUC | 76.15% |
| Top-1 Candidate Accuracy | 80.09% |

## 6. Observations

1. The model achieved 94.61% validation accuracy.
2. Precision of 77.76% indicates that most candidates predicted as positive were actually positive.
3. Recall of 70.75% indicates that the model missed a proportion of actual positive candidates.
4. ROC-AUC of 93.54% and PR-AUC of 76.15% indicate reasonable classification performance.
5. Top-1 candidate accuracy of 80.09% shows that the correct candidate was ranked first for approximately 80.09% of labeled observations.
6. KNN achieved a very short training time of 0.08 seconds and a validation prediction time of 1.32 seconds.
7. Unlike the previous tree-based models, KNN requires feature scaling because it uses distance calculations.
8. The model was trained using a 100,000-row stratified sample rather than the full training dataset.

## 7. Limitations

1. The model was trained on a subset of the available training data.
2. KNN prediction cost can increase significantly as the training dataset grows.
3. Performance depends on the choice of distance metric, number of neighbors, and feature scaling.
4. KNN does not provide native feature importance scores.

## 8. Conclusion

The KNN model provides a distance-based baseline for the PathSnap candidate classification task.

It achieved 94.61% validation accuracy but lower recall, F1-score, ROC-AUC, PR-AUC, and top-1 candidate accuracy compared with the XGBoost baseline.

Its short training and prediction times are useful advantages for the sampled experiment. However, its performance and scalability should be considered before selecting it as the final model.

KNN is retained as the fourth baseline model for comparison.

## 9. Saved Artifacts

- Trained model: `models/artifacts/knn_model.joblib`
- Evaluation metrics: `reports/models/knn_metrics.json`
- Training script: `src/models/train_knn.py`