# Shared candidate dataset contract

Use derived/candidate_dataset.csv and its manifest as the single candidate dataset for all five models. Regenerate it only with candidate_dataset.py; do not build per-model candidate datasets.

## Columns

- Identifiers: trip_id, observation_index, candidate_arc_id
- Features, identical for every model: candidate_rank, distance_m, heading_difference_deg, road_length_m, gps_speed_mps
- Target: target
- Split metadata: dataset_split

The observation and arc identifiers refer to original zero-based source row positions. Candidate rank is 1-based and sorted by point-to-segment distance, then arc ID.

## Target definition

For each observation, consider only generated candidates within the configured radius that also occur in that trip's supplied .route. The nearest such candidate is the positive class (target=1); all other candidates for the same observation are negative (target=0). Distance ties use the first occurrence in the ordered reference route, then arc ID. If no route arc occurs in the candidate set, omit that observation rather than making all candidates negative.

This is a candidate-level proxy label. The supplied route is an ordered arc sequence without per-GPS timestamps, so this definition does not claim exact timestamp-to-route alignment. It avoids assuming that GPS row indices align with route row indices.

## Leakage-safe splits

The 80 train trips are split by trip ID, using seed 42: 64 train, 8 validation, and 8 held-out test trips. All 10 supplied student_test trips are marked student_test_dev; they must not be used to fit models or learned preprocessing. Use dataset_split for every model and keep all rows from each trip together.

## Missing features

Blank feature cells mean the feature is unavailable (for example, heading or speed at the first observation). Fit any imputation or other learned preprocessing using the train split only, then apply it to validation/test/dev.
