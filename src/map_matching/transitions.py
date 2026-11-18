
import heapq
from collections import OrderedDict

import road_graph


class TransitionScorer:
    """
    Calculate HMM transition scores using directed road-network distances.
    """

    def __init__(
        self,
        graph,
        beta=50.0,
        unreachable_penalty=20.0,
        cache_size=128,
    ):
        if beta <= 0:
            raise ValueError("beta must be positive")

        self.graph = graph
        self.beta = float(beta)
        self.unreachable_penalty = float(unreachable_penalty)
        self.cache_size = cache_size

        # Cache shortest-path distances by source node.
        self.distance_cache = OrderedDict()

    def _position_fraction(self, observation, segment):
        """Project GPS observation onto a road segment."""
        point = road_graph._project(
            observation.longitude,
            observation.latitude,
            self.graph.reference_latitude,
        )

        ax, ay = segment.start_xy
        dx = segment.end_xy[0] - ax
        dy = segment.end_xy[1] - ay

        length_squared = dx * dx + dy * dy

        if length_squared == 0:
            return 0.0

        fraction = (
            (point[0] - ax) * dx +
            (point[1] - ay) * dy
        ) / length_squared

        return max(0.0, min(1.0, fraction))

    def _shortest_paths_from(self, source_node):
        """Run directed Dijkstra and cache distances from a source node."""

        if source_node in self.distance_cache:
            self.distance_cache.move_to_end(source_node)
            return self.distance_cache[source_node]

        distances = {source_node: 0.0}
        queue = [(0.0, source_node)]

        while queue:
            current_distance, node = heapq.heappop(queue)

            if current_distance > distances[node]:
                continue

            for next_node, arc_id in self.graph.outgoing(node):
                segment = self.graph.segments[arc_id]
                new_distance = current_distance + segment.length_m

                if new_distance < distances.get(next_node, float("inf")):
                    distances[next_node] = new_distance
                    heapq.heappush(
                        queue,
                        (new_distance, next_node),
                    )

        self.distance_cache[source_node] = distances
        self.distance_cache.move_to_end(source_node)

        if len(self.distance_cache) > self.cache_size:
            self.distance_cache.popitem(last=False)

        return distances

    def network_distance(
        self,
        previous_observation,
        previous_candidate,
        current_observation,
        current_candidate,
    ):
        """
        Calculate directed distance between two candidate positions.

        Returns None when no directed path exists.
        """

        previous_segment = self.graph.segments.get(
            previous_candidate.arc_id
        )
        current_segment = self.graph.segments.get(
            current_candidate.arc_id
        )

        if previous_segment is None or current_segment is None:
            return None

        previous_fraction = self._position_fraction(
            previous_observation,
            previous_segment,
        )

        current_fraction = self._position_fraction(
            current_observation,
            current_segment,
        )

        # Same road segment: forward movement.
        if (
            previous_candidate.arc_id == current_candidate.arc_id
            and current_fraction >= previous_fraction
        ):
            return (
                current_fraction - previous_fraction
            ) * previous_segment.length_m

        # Remaining distance on previous segment.
        remaining_distance = (
            1.0 - previous_fraction
        ) * previous_segment.length_m

        # Distance along the current segment before reaching GPS position.
        current_prefix_distance = (
            current_fraction * current_segment.length_m
        )

        # Find directed path between the two road segments.
        distances = self._shortest_paths_from(
            previous_segment.to_node_id
        )

        middle_distance = distances.get(
            current_segment.from_node_id
        )

        if middle_distance is None:
            return None

        return (
            remaining_distance
            + middle_distance
            + current_prefix_distance
        )

    def score(
        self,
        previous_observation,
        previous_candidate,
        current_observation,
        current_candidate,
    ):
        """
        Return a log transition score.

        Uses an exponential distance-difference model:
            log_score = -abs(network_distance - gps_distance) / beta
        """

        network_distance = self.network_distance(
            previous_observation,
            previous_candidate,
            current_observation,
            current_candidate,
        )

        if network_distance is None:
            return -self.unreachable_penalty

        previous_xy = road_graph._project(
            previous_observation.longitude,
            previous_observation.latitude,
            self.graph.reference_latitude,
        )

        current_xy = road_graph._project(
            current_observation.longitude,
            current_observation.latitude,
            self.graph.reference_latitude,
        )

        gps_distance = road_graph._distance(
            previous_xy,
            current_xy,
        )

        difference = abs(network_distance - gps_distance)

        return -difference / self.beta
