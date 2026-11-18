"""Reusable loader for the PathSnap tab-separated trip files.

Row IDs are zero-based positions in the original source file. Invalid rows are
reported and skipped without renumbering later nodes, arcs, GPS observations,
or route entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path


@dataclass(frozen=True)
class GPSObservation:
    observation_index: int
    longitude: float
    latitude: float
    timestamp: float


@dataclass
class TripData:
    trip_id: str
    split: str
    gps: list[GPSObservation] = field(default_factory=list)
    nodes: dict[int, tuple[float, float]] = field(default_factory=dict)
    arcs: dict[int, tuple[int, int]] = field(default_factory=dict)
    route_arc_ids: list[int] = field(default_factory=list)
    rejected_records: list[dict] = field(default_factory=list)


def _reject(trip: TripData, file_type: str, row_index: int, reason: str) -> None:
    trip.rejected_records.append({
        "file_type": file_type,
        "row_index": row_index,
        "reason": reason,
    })


def _read_lines(path: Path):
    # Keep blank lines so row indexes remain identical to source file indexes.
    with path.open("r", encoding="utf-8-sig") as stream:
        yield from enumerate(stream)


def _fields(line: str) -> list[str]:
    return line.rstrip("\r\n").split("\t")


def _number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def _integer(text: str) -> int:
    # Reject values such as 12.0 rather than silently coercing them.
    value = int(text)
    if str(value) != text.strip():
        raise ValueError("not an integer")
    return value


def load_trip(trip_dir: str | Path, split: str | None = None) -> TripData:
    """Load one trip directory containing matching .track/.nodes/.arcs/.route files."""
    directory = Path(trip_dir)
    trip_id = directory.name
    split_name = split or directory.parent.name
    trip = TripData(trip_id=trip_id, split=split_name)

    required = {ext: directory / f"{trip_id}{ext}"
                for ext in (".track", ".nodes", ".arcs", ".route")}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Trip {trip_id} is missing files: {', '.join(missing)}")

    for row_index, line in _read_lines(required[".track"]):
        try:
            fields = _fields(line)
            if len(fields) != 3:
                raise ValueError("expected 3 tab-separated fields")
            longitude, latitude, timestamp = map(_number, fields)
            if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                raise ValueError("GPS coordinate outside longitude/latitude bounds")
            trip.gps.append(GPSObservation(row_index, longitude, latitude, timestamp))
        except (ValueError, OverflowError) as error:
            _reject(trip, ".track", row_index, str(error))

    for row_index, line in _read_lines(required[".nodes"]):
        try:
            fields = _fields(line)
            if len(fields) != 2:
                raise ValueError("expected 2 tab-separated fields")
            longitude, latitude = map(_number, fields)
            if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                raise ValueError("node coordinate outside longitude/latitude bounds")
            trip.nodes[row_index] = (longitude, latitude)
        except (ValueError, OverflowError) as error:
            _reject(trip, ".nodes", row_index, str(error))

    # Parse first, then validate endpoints against original node row IDs.
    raw_arcs: dict[int, tuple[int, int]] = {}
    for row_index, line in _read_lines(required[".arcs"]):
        try:
            fields = _fields(line)
            if len(fields) != 2:
                raise ValueError("expected 2 tab-separated fields")
            raw_arcs[row_index] = (_integer(fields[0]), _integer(fields[1]))
        except (ValueError, OverflowError) as error:
            _reject(trip, ".arcs", row_index, str(error))

    for arc_id, (from_node, to_node) in raw_arcs.items():
        if from_node not in trip.nodes or to_node not in trip.nodes:
            _reject(trip, ".arcs", arc_id, "arc endpoint references a missing or invalid node")
            continue
        trip.arcs[arc_id] = (from_node, to_node)

    for row_index, line in _read_lines(required[".route"]):
        try:
            fields = _fields(line)
            if len(fields) != 1:
                raise ValueError("expected 1 tab-separated field")
            arc_id = _integer(fields[0])
            if arc_id not in trip.arcs:
                raise ValueError("route references a missing or invalid arc")
            trip.route_arc_ids.append(arc_id)
        except (ValueError, OverflowError) as error:
            _reject(trip, ".route", row_index, str(error))

    return trip


def load_dataset(dataset_root: str | Path) -> dict[str, list[TripData]]:
    """Load train and student_test separately from a dataset root directory."""
    root = Path(dataset_root)
    splits: dict[str, list[TripData]] = {}
    for split in ("train", "student_test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        trips = []
        for trip_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            trips.append(load_trip(trip_dir, split=split))
        splits[split] = trips
    if not splits:
        raise FileNotFoundError(f"No train or student_test directories found under {root}")
    return splits
