# PathSnap data pipeline

This project prepares GPS trips and nearby road segments for map matching. The same candidate dataset is meant to be used by all five models, so each model sees the same examples, features, labels, and trip splits.

## What is in this repository

- data_loader.py reads each trip's GPS points, road nodes, road arcs, and reference route. It skips malformed or invalid    records and keeps the original row IDs.
- road_graph.py builds a directed graph from the road arcs and finds nearby road candidates for each GPS point.
- candidate_dataset.py combines those candidates into one labeled CSV.
- candidate_dataset_spec.md describes the shared columns, target, and split rules.
- data/raw/candidate_dataset.csv.gz is the processed candidate dataset.
- data/raw/candidate_dataset_manifest.json records trip assignments and row counts.
- data/raw/Student folder.zip is the supplied raw dataset archive.

## What the processing does

For each trip, the loader reads:

- GPS observations: longitude, latitude, and timestamp.
- Road nodes: longitude and latitude. The row number is the original node ID.
- Road arcs: directed connections between nodes. The row number is the original arc ID.
- Reference route: the ordered arc IDs for that trip.

The road graph keeps the original node and arc IDs. For each GPS observation, candidate generation finds road segments within 50 metres and calculates:

- Candidate rank by distance.
- Distance from the GPS point to the road segment.
- Difference between GPS heading and directed road heading, when available.
- Road segment length.
- GPS speed, when timestamps allow it.

The CSV keeps the trip ID, original GPS observation index, and original candidate arc ID. Blank feature cells mean a value was unavailable, such as heading or speed for the first observation. Any imputation must be learned from the training split only.

## What the target means

The reference route is a sequence of road arc IDs, but it has no GPS timestamps. We therefore do not pair GPS and route rows by their row numbers.

For each GPS observation, we look at its nearby candidates that also appear in that trip's reference route. The closest such candidate gets target=1; the other candidates for that observation get target=0. If none of the nearby candidates appears in the reference route, that observation is left out of the labeled dataset.

This is a practical proxy label, not a claim that the route tells us the exact correct arc at every GPS timestamp. All five models should use this same target definition.

## Trip splits

All rows from one trip stay in one split:

- train: 64 trips from the supplied 80-trip training partition.
- validation: 8 trips from that training partition.
- test: 8 held-out trips from that training partition.
- student_test_dev: all 10 supplied student-test trips, for development evaluation only. Do not use these trips to fit models or preprocessing.

The 80 training trips are sorted, shuffled with seed 42, and split 80% / 10% / 10%. The exact trip assignments are in the manifest.

## Processed data summary

At the 50-metre candidate radius, the dataset contains 2,044,065 candidate rows from 90 trips:

| Split | Candidate rows | Labeled GPS observations |
|---|---:|---:|
| train | 1,536,611 | 155,243 |
| validation | 157,591 | 17,158 |
| test | 140,344 | 13,462 |
| student_test_dev | 209,519 | 19,133 |

Some GPS observations are omitted when they have no nearby candidate from the reference route. Counts are also recorded in the manifest.

## Read the processed dataset

Python's pandas can read the compressed CSV directly. For example:

    import pandas as pd

    data = pd.read_csv("data/raw/candidate_dataset.csv.gz")
    train = data[data["dataset_split"] == "train"]
    validation = data[data["dataset_split"] == "validation"]
    test = data[data["dataset_split"] == "test"]

## Regenerate it

Extract the supplied dataset archive, then pass the extracted folder to the builder:

    python candidate_dataset.py "Student folder" --output "data/raw/candidate_dataset.csv.gz" --manifest "data/raw/candidate_dataset_manifest.json" --radius-m 50 --seed 42

The builder uses Python's standard library. Keep the extracted raw dataset out of Git; the repository ignore rules exclude extracted raw data.
