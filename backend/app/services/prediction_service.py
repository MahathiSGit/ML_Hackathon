
from pathlib import Path

from backend.app.services.inference_loader import load_inference_trip
from src.map_matching.hmm_matcher import HMMMapMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load the trained model once when the service is initialized.
matcher = HMMMapMatcher(project_root=PROJECT_ROOT)


def predict_trip(trip_id: str):
    """
    Generate a map-matched route for a requested trip.

    Ground-truth .route files are never loaded here.
    """

    if not trip_id.isdigit():
        raise ValueError("Invalid trip ID.")

    trip_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "validation_trips"
        / "train"
        / trip_id
    )

    if not trip_dir.is_dir():
        raise FileNotFoundError(
            f"Trip {trip_id} was not found."
        )

    trip = load_inference_trip(
        trip_dir,
        split="validation",
    )

    result = matcher.match_trip(trip)

    return {
        "trip_id": result["trip_id"],
        "gps_observations": result["gps_observations"],
        "predicted_observations": result["predicted_observations"],
        "missing_predictions": result["missing_predictions"],
        "reconstructed_route": result["reconstructed_route"],
        "unique_predicted_arcs": result["unique_predicted_arcs"],
    }
