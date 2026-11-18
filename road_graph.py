"""Directed road graph and spatially indexed GPS-to-road candidates."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

try:
    from .data_loader import GPSObservation, TripData
except ImportError:
    from data_loader import GPSObservation, TripData


EARTH_RADIUS_M = 6_371_008.8


def _project(longitude: float, latitude: float, reference_latitude: float) -> tuple[float, float]:
    """Local equirectangular projection for trip-scale distance calculations."""
    x = math.radians(longitude) * EARTH_RADIUS_M * math.cos(math.radians(reference_latitude))
    y = math.radians(latitude) * EARTH_RADIUS_M
    return x, y


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bearing(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _heading_difference(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class RoadSegment:
    arc_id: int
    from_node_id: int
    to_node_id: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    length_m: float
    heading_deg: float | None


@dataclass(frozen=True)
class Candidate:
    observation_index: int
    arc_id: int
    distance_m: float
    heading_difference_deg: float | None
    road_length_m: float
    gps_speed_mps: float | None


class RoadGraph:
    """Directed graph keyed by original node and arc row IDs."""

    def __init__(self, trip: TripData):
        if trip.gps:
            self.reference_latitude = sum(p.latitude for p in trip.gps) / len(trip.gps)
        elif trip.nodes:
            self.reference_latitude = sum(p[1] for p in trip.nodes.values()) / len(trip.nodes)
        else:
            self.reference_latitude = 0.0

        self.trip_id = trip.trip_id
        self.split = trip.split
        self.nodes = {
            node_id: _project(lon, lat, self.reference_latitude)
            for node_id, (lon, lat) in trip.nodes.items()
        }
        self.adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.segments: dict[int, RoadSegment] = {}
        for arc_id, (from_id, to_id) in trip.arcs.items():
            if from_id not in self.nodes or to_id not in self.nodes:
                continue
            start, end = self.nodes[from_id], self.nodes[to_id]
            segment = RoadSegment(
                arc_id, from_id, to_id, start, end,
                _distance(start, end), _bearing(start, end)
            )
            self.segments[arc_id] = segment
            self.adjacency[from_id].append((to_id, arc_id))

    def outgoing(self, node_id: int) -> tuple[tuple[int, int], ...]:
        """Return (to_node_id, arc_id) pairs for directed outgoing arcs."""
        return tuple(self.adjacency.get(node_id, ()))


class CandidateGenerator:
    """Nearby directed segments via a sampled-point spatial hash."""

    def __init__(self, graph: RoadGraph, radius_m: float = 50.0):
        if not math.isfinite(radius_m) or radius_m <= 0:
            raise ValueError("radius_m must be a positive finite number")
        self.graph = graph
        self.radius_m = float(radius_m)
        self._cell_size = self.radius_m
        sample_step = self._cell_size / 2.0
        self._query_radius = self.radius_m + sample_step / 2.0
        self._grid: dict[tuple[int, int], set[int]] = defaultdict(set)

        # Sample each segment at <= half-cell spacing. Any segment within the
        # search radius then has a sample within radius + half-cell/2.
        for arc_id, segment in graph.segments.items():
            intervals = max(1, math.ceil(segment.length_m / sample_step))
            dx = segment.end_xy[0] - segment.start_xy[0]
            dy = segment.end_xy[1] - segment.start_xy[1]
            for i in range(intervals + 1):
                fraction = i / intervals
                self._grid[self._cell(
                    segment.start_xy[0] + fraction * dx,
                    segment.start_xy[1] + fraction * dy,
                )].add(arc_id)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / self._cell_size), math.floor(y / self._cell_size)

    def _nearby_arc_ids(self, point: tuple[float, float]) -> set[int]:
        x, y = point
        extent = self._query_radius
        low_x = math.floor((x - extent) / self._cell_size)
        high_x = math.floor((x + extent) / self._cell_size)
        low_y = math.floor((y - extent) / self._cell_size)
        high_y = math.floor((y + extent) / self._cell_size)
        found: set[int] = set()
        for gx in range(low_x, high_x + 1):
            for gy in range(low_y, high_y + 1):
                found.update(self._grid.get((gx, gy), ()))
        return found

    @staticmethod
    def _point_segment_distance(
        point: tuple[float, float], segment: RoadSegment
    ) -> float:
        ax, ay = segment.start_xy
        dx = segment.end_xy[0] - ax
        dy = segment.end_xy[1] - ay
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            return _distance(point, segment.start_xy)
        fraction = ((point[0] - ax) * dx + (point[1] - ay) * dy) / length_squared
        fraction = min(1.0, max(0.0, fraction))
        nearest = (ax + fraction * dx, ay + fraction * dy)
        return _distance(point, nearest)

    def candidates_for_observation(
        self,
        observation: GPSObservation,
        heading_deg: float | None = None,
        gps_speed_mps: float | None = None,
    ) -> list[Candidate]:
        point = _project(
            observation.longitude, observation.latitude,
            self.graph.reference_latitude
        )
        result = []
        for arc_id in self._nearby_arc_ids(point):
            segment = self.graph.segments[arc_id]
            distance = self._point_segment_distance(point, segment)
            if distance <= self.radius_m:
                result.append(Candidate(
                    observation_index=observation.observation_index,
                    arc_id=arc_id,
                    distance_m=distance,
                    heading_difference_deg=_heading_difference(
                        heading_deg, segment.heading_deg
                    ),
                    road_length_m=segment.length_m,
                    gps_speed_mps=gps_speed_mps,
                ))
        result.sort(key=lambda candidate: (candidate.distance_m, candidate.arc_id))
        return result


def candidates_for_trip(
    trip: TripData, radius_m: float = 50.0
) -> tuple[RoadGraph, dict[int, list[Candidate]]]:
    """Build a graph and candidates keyed by each original GPS row index."""
    graph = RoadGraph(trip)
    generator = CandidateGenerator(graph, radius_m=radius_m)
    observations = sorted(trip.gps, key=lambda point: point.observation_index)
    by_index = {point.observation_index: [] for point in observations}

    for position, observation in enumerate(observations):
        heading = speed = None
        if position > 0:
            previous = observations[position - 1]
            delta_seconds = observation.timestamp - previous.timestamp
            start = _project(
                previous.longitude, previous.latitude, graph.reference_latitude
            )
            end = _project(
                observation.longitude, observation.latitude, graph.reference_latitude
            )
            traveled = _distance(start, end)
            if delta_seconds > 0:
                speed = traveled / delta_seconds
                heading = _bearing(start, end)
        by_index[observation.observation_index] = generator.candidates_for_observation(
            observation, heading_deg=heading, gps_speed_mps=speed
        )
    return graph, by_index


# Descriptive alias for callers that prefer the explicit operation name.
build_graph_and_candidates = candidates_for_trip
