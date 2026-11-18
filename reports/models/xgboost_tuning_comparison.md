# XGBoost: Baseline vs Bayesian Optimized

| Metric | Baseline XGBoost | Tuned XGBoost |
|---|---:|---:|
| accuracy | 0.9600 | 0.9563 |
| precision | 0.8366 | 0.7670 |
| recall | 0.7862 | 0.8601 |
| f1_score | 0.8106 | 0.8109 |
| roc_auc | 0.9808 | 0.9809 |
| pr_auc | 0.8753 | 0.8747 |
| top1_candidate_accuracy | 0.8628 | 0.8672 |

## Best Optimization Configuration

```json
{
    "n_estimators": 302,
    "max_depth": 4,
    "learning_rate": 0.023231731978492907,
    "subsample": 0.9795542149013333,
    "colsample_bytree": 0.9862528132298237,
    "min_child_weight": 9,
    "gamma": 1.5230688458668533,
    "reg_alpha": 7.569183361880229e-08,
    "reg_lambda": 0.8768125777016167,
    "scale_pos_weight": 1.768335486227123
}
```