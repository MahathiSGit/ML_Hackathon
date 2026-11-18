"""Build the single shared candidate-level PathSnap dataset."""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

try:
    from .data_loader import load_dataset
    from .road_graph import candidates_for_trip
except ImportError:
    from data_loader import load_dataset
    from road_graph import candidates_for_trip


FEATURE_COLUMNS = (
    "candidate_rank",
    "distance_m",
    "heading_difference_deg",
    "road_length_m",
    "gps_speed_mps",
)
IDENTIFIER_COLUMNS = ("trip_id", "observation_index", "candidate_arc_id")
TARGET_COLUMN = "target"
TARGET_DEFINITION = (
    "For each GPS observation, among its candidate arcs within the configured "
    "search radius that occur in the trip's supplied reference route, target=1 "
    "for the nearest such arc and target=0 for the other candidates. Distance "
    "ties are broken by first occurrence in the ordered reference route, then "
    "by arc ID. An observation with no in-radius reference-route candidate is "
    "omitted rather than assigned a false all-negative label. This is a "
    "candidate-level proxy label: the route has no per-GPS timestamps, so it "
    "does not claim exact timestamp-to-route alignment."
)


def _training_trip_splits(trip_ids: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(trip_ids)
    random.Random(seed).shuffle(ordered)
    train_end = int(len(ordered) * 0.8)
    validation_end = train_end + int(len(ordered) * 0.1)
    return {
        **{trip_id: "train" for trip_id in ordered[:train_end]},
        **{trip_id: "validation" for trip_id in ordered[train_end:validation_end]},
        **{trip_id: "test" for trip_id in ordered[validation_end:]},
    }


def build_candidate_dataset(
    dataset_root: str | Path,
    output_csv: str | Path,
    manifest_json: str | Path,
    radius_m: float = 50.0,
    seed: int = 42,
) -> dict:
    """Build one CSV for all models, with trip-level split labels and manifest."""
    loaded = load_dataset(dataset_root)
    train_trips = loaded.get("train", [])
    student_test_trips = loaded.get("student_test", [])
    trip_splits = _training_trip_splits([trip.trip_id for trip in train_trips], seed)
    trip_splits.update({
        trip.trip_id: "student_test_dev" for trip in student_test_trips
    })

    output_path = Path(output_csv)
    manifest_path = Path(manifest_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        *IDENTIFIER_COLUMNS,
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
        "dataset_split",
    ]
    row_counts: Counter[str] = Counter()
    observation_counts: Counter[str] = Counter()
    dropped_observations: Counter[str] = Counter()
    trip_summary: dict[str, dict] = {}

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        for source_split in ("train", "student_test"):
            for trip in sorted(loaded.get(source_split, []), key=lambda item: item.trip_id):
                dataset_split = trip_splits[trip.trip_id]
                _, candidates_by_observation = candidates_for_trip(trip, radius_m)
                route_order: dict[int, int] = {}
                for route_position, arc_id in enumerate(trip.route_arc_ids):
                    route_order.setdefault(arc_id, route_position)
                trip_rows = 0
                trip_labeled = 0
                trip_dropped = 0

                for observation_index, candidates in candidates_by_observation.items():
                    route_candidates = [
                        item for item in candidates if item.arc_id in route_order
                    ]
                    if not route_candidates:
                        trip_dropped += 1
                        dropped_observations[dataset_split] += 1
                        continue
                    best = min(
                        route_candidates,
                        key=lambda item: (
                            item.distance_m, route_order[item.arc_id], item.arc_id
                        ),
                    )
                    trip_labeled += 1
                    observation_counts[dataset_split] += 1

                    for rank, candidate in enumerate(candidates, start=1):
                        writer.writerow({
                            "trip_id": trip.trip_id,
                            "observation_index": observation_index,
                            "candidate_arc_id": candidate.arc_id,
                            "candidate_rank": rank,
                            "distance_m": candidate.distance_m,
                            "heading_difference_deg": candidate.heading_difference_deg,
                            "road_length_m": candidate.road_length_m,
                            "gps_speed_mps": candidate.gps_speed_mps,
                            "target": int(candidate.arc_id == best.arc_id),
                            "dataset_split": dataset_split,
                        })
                        row_counts[dataset_split] += 1
                        trip_rows += 1

                trip_summary[trip.trip_id] = {
                    "source_partition": source_split,
                    "dataset_split": dataset_split,
                    "gps_observations": len(trip.gps),
                    "labeled_observations": trip_labeled,
                    "omitted_observations_without_route_candidate": trip_dropped,
                    "candidate_rows": trip_rows,
                    "rejected_input_records": len(trip.rejected_records),
                }

    manifest = {
        "dataset_root": str(dataset_root),
        "candidate_csv": str(output_path),
        "search_radius_m": radius_m,
        "random_seed": seed,
        "train_partition_split_method": (
            "Sort train trip IDs, shuffle with random.Random(seed), then allocate "
            "first 80% to train, next 10% to validation, remainder to test."
        ),
        "student_test_policy": (
            "All provided student_test trips are labeled student_test_dev and "
            "must not be used to fit models or fit preprocessing."
        ),
        "target_column": TARGET_COLUMN,
        "target_definition": TARGET_DEFINITION,
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "missing_feature_values": (
            "Blank CSV cells represent unavailable heading or speed (for example, "
            "the first observation or nonpositive timestamp difference). Fit any "
            "imputation using only the training trip split."
        ),
        "counts_by_split": {
            split: {
                "candidate_rows": row_counts[split],
                "labeled_observations": observation_counts[split],
                "omitted_observations_without_route_candidate": dropped_observations[split],
            }
            for split in ("train", "validation", "test", "student_test_dev")
        },
        "trip_assignments": trip_splits,
        "trips": trip_summary,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Directory containing train/ and student_test/")
    parser.add_argument("--output", default="derived/candidate_dataset.csv")
    parser.add_argument("--manifest", default="derived/candidate_dataset_manifest.json")
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build_candidate_dataset(
        args.dataset_root, args.output, args.manifest, args.radius_m, args.seed
    )
    print(json.dumps(result["counts_by_split"], indent=2))
