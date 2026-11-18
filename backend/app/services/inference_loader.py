
from pathlib import Path
from dataclasses import dataclass

from data_loader import GPSObservation, TripData


@dataclass
class InferenceTrip:
    trip_id: str
    split: str
    gps: list
    nodes: dict
    arcs: dict


def load_inference_trip(trip_dir: Path, split="student_test"):
    """
    Load GPS and road-network data without reading the .route file.
    """

    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"

    for file_path in [track_file, nodes_file, arcs_file]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required trip file not found: {file_path}"
            )

    gps = []

    with track_file.open("r") as file:
        for index, line in enumerate(file):
            values = line.split()

            if len(values) != 3:
                continue

            longitude, latitude, timestamp = map(float, values)

            gps.append(
                GPSObservation(
                    observation_index=index,
                    longitude=longitude,
                    latitude=latitude,
                    timestamp=timestamp,
                )
            )

    nodes = {}

    with nodes_file.open("r") as file:
        for index, line in enumerate(file):
            values = line.replace(",", " ").split()

            if len(values) != 2:
                continue

            longitude, latitude = map(float, values)
            nodes[index] = (longitude, latitude)

    arcs = {}

    with arcs_file.open("r") as file:
        for index, line in enumerate(file):
            values = line.replace(",", " ").split()

            if len(values) != 2:
                continue

            from_id, to_id = map(int, values)
            arcs[index] = (from_id, to_id)

    return InferenceTrip(
        trip_id=trip_id,
        split=split,
        gps=gps,
        nodes=nodes,
        arcs=arcs,
    )
