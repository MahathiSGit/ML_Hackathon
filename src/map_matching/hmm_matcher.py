
from pathlib import Path

import pandas as pd

from road_graph import candidates_for_trip
from src.map_matching.emissions import XGBoostEmissionScorer
from src.map_matching.transitions import TransitionScorer
from src.map_matching.viterbi import ViterbiDecoder


class HMMMapMatcher:
    """Complete XGBoost + HMM map-matching pipeline."""

    def __init__(
        self,
        radius_m=50.0,
        beta=50.0,
        project_root=None,
    ):
        self.radius_m = radius_m

        self.emission_scorer = XGBoostEmissionScorer(
            project_root=project_root
        )

        self.decoder = ViterbiDecoder()
        self.beta = beta

    def match_trip(self, trip):
        """
        Map-match a trip without accessing its reference route.

        Returns predictions and diagnostic information.
        """

        observations = sorted(
            trip.gps,
            key=lambda p: p.observation_index,
        )

        # Step 1: Generate road candidates.
        graph, candidates = candidates_for_trip(
            trip,
            radius_m=self.radius_m,
        )

        # Step 2: Calculate XGBoost emission scores.
        emissions = {}

        for observation_index, candidate_list in candidates.items():

            if not candidate_list:
                emissions[observation_index] = []
                continue

            rows = pd.DataFrame([
                {
                    "candidate_rank": rank,
                    "distance_m": candidate.distance_m,
                    "heading_difference_deg":
                        candidate.heading_difference_deg,
                    "road_length_m": candidate.road_length_m,
                    "gps_speed_mps": candidate.gps_speed_mps,
                }
                for rank, candidate in enumerate(
                    candidate_list,
                    start=1,
                )
            ])

            emissions[observation_index] = (
                self.emission_scorer.score(rows)
            )

        # Step 3: Initialize transition scorer.
        transition_scorer = TransitionScorer(
            graph,
            beta=self.beta,
        )

        # Step 4: Decode the best candidate sequence.
        predicted_candidates = self.decoder.decode(
            observations,
            candidates,
            emissions,
            transition_scorer,
        )

        # Preserve one prediction slot per GPS observation.
        predicted_arc_ids = [
            candidate.arc_id if candidate is not None else None
            for candidate in predicted_candidates
        ]

        # Remove consecutive duplicate arcs for route representation.
        reconstructed_route = []

        for arc_id in predicted_arc_ids:
            if arc_id is None:
                continue

            if (
                not reconstructed_route
                or reconstructed_route[-1] != arc_id
            ):
                reconstructed_route.append(arc_id)

        return {
            "trip_id": trip.trip_id,
            "predicted_candidates": predicted_candidates,
            "predicted_arc_ids": predicted_arc_ids,
            "reconstructed_route": reconstructed_route,
            "gps_observations": len(observations),
            "predicted_observations": sum(
                arc_id is not None for arc_id in predicted_arc_ids
            ),
            "missing_predictions": sum(
                arc_id is None for arc_id in predicted_arc_ids
            ),
            "unique_predicted_arcs": len(set(
                arc_id for arc_id in predicted_arc_ids
                if arc_id is not None
            )),
            "shortest_path_cache_entries": len(
                transition_scorer.distance_cache
            ),
        }
