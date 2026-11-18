from pathlib import Path
from collections import Counter

import folium

from data_loader import load_trip
from road_graph import candidates_for_trip
from backend.app.services.prediction_service import predict_trip


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TRIP_ID = "00000005"
SEARCH_RADIUS_M = 50

TRIP_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "validation_trips"
    / "train"
    / TRIP_ID
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / f"trip_{TRIP_ID}_map.html"


# =====================================================
# 2. LOAD TRIP
# =====================================================

def generate_map():

    print("\nLoading trip:", TRIP_ID)

    if not TRIP_DIR.exists():
        raise FileNotFoundError(TRIP_DIR)

    trip = load_trip(TRIP_DIR, split="validation")

    print("GPS observations:", len(trip.gps))
    print("Road nodes:", len(trip.nodes))
    print("Road arcs:", len(trip.arcs))


    # =================================================
    # 3. GENERATE ALL POTENTIAL ROAD CANDIDATES
    # =================================================

    print("\nGenerating potential road candidates...")

    graph, candidates_by_observation = candidates_for_trip(
        trip,
        radius_m=SEARCH_RADIUS_M
    )

    # Count how frequently each road arc appears
    # as a candidate for GPS observations.

    candidate_frequency = Counter()

    for candidates in candidates_by_observation.values():
        for candidate in candidates:
            candidate_frequency[candidate.arc_id] += 1

    potential_arc_ids = set(candidate_frequency.keys())

    print("GPS observations:", len(candidates_by_observation))
    print("Unique potential road arcs:", len(potential_arc_ids))


    # =================================================
    # 4. RUN XGBOOST + HMM
    # =================================================

    print("\nRunning XGBoost + HMM...")

    result = predict_trip(TRIP_ID)

    predicted_arc_ids = result["reconstructed_route"]

    predicted_arc_ids = list(dict.fromkeys(predicted_arc_ids))

    print("Predicted unique road arcs:", len(predicted_arc_ids))


    # =================================================
    # 5. PREPARE GPS COORDINATES
    # =================================================

    gps_coordinates = [
        (point.latitude, point.longitude)
        for point in sorted(
            trip.gps,
            key=lambda point: point.observation_index
        )
    ]

    if not gps_coordinates:
        raise ValueError("No GPS observations found.")

    center_lat = sum(p[0] for p in gps_coordinates) / len(gps_coordinates)
    center_lon = sum(p[1] for p in gps_coordinates) / len(gps_coordinates)


    # =================================================
    # 6. CREATE MAP
    # =================================================

    print("\nCreating interactive map...")

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )

    # Esri basemap avoids the OpenStreetMap 403 issue.

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Street_Map/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles &copy; Esri",
        name="Esri World Street Map",
        overlay=False,
        control=True
    ).add_to(m)


    # =================================================
    # 7. POTENTIAL ROAD CANDIDATES LAYER
    # =================================================

    print("Drawing potential road candidates...")

    candidate_layer = folium.FeatureGroup(
        name="Potential Road Candidates",
        show=True
    )

    candidate_coordinates = []

    for arc_id in potential_arc_ids:

        if arc_id not in trip.arcs:
            continue

        from_node, to_node = trip.arcs[arc_id]

        if (
            from_node not in trip.nodes
            or to_node not in trip.nodes
        ):
            continue

        from_lon, from_lat = trip.nodes[from_node]
        to_lon, to_lat = trip.nodes[to_node]

        start = (from_lat, from_lon)
        end = (to_lat, to_lon)

        frequency = candidate_frequency[arc_id]

        # Draw alternative road candidates.
        # More frequently considered candidates are
        # slightly more visible.

        opacity = min(0.65, 0.25 + frequency * 0.015)

        folium.PolyLine(
            locations=[start, end],
            color="#E9A23B",
            weight=2,
            opacity=opacity,
            tooltip=(
                f"Potential Candidate Arc: {arc_id}"
                f" | Selected for {frequency} GPS observations"
            )
        ).add_to(candidate_layer)

        candidate_coordinates.extend([start, end])

    candidate_layer.add_to(m)


    # =================================================
    # 8. PREDICTED ROUTE LAYER
    # =================================================

    print("Drawing final predicted route...")

    predicted_layer = folium.FeatureGroup(
        name="Final Predicted Route (XGBoost + HMM)",
        show=True
    )

    predicted_coordinates = []

    valid_predicted_arcs = 0

    for arc_id in predicted_arc_ids:

        if arc_id not in trip.arcs:
            continue

        from_node, to_node = trip.arcs[arc_id]

        if (
            from_node not in trip.nodes
            or to_node not in trip.nodes
        ):
            continue

        from_lon, from_lat = trip.nodes[from_node]
        to_lon, to_lat = trip.nodes[to_node]

        start = (from_lat, from_lon)
        end = (to_lat, to_lon)

        # Dark outline for visual contrast.

        folium.PolyLine(
            locations=[start, end],
            color="#8B0000",
            weight=9,
            opacity=0.75
        ).add_to(predicted_layer)

        # Main predicted road segment.

        folium.PolyLine(
            locations=[start, end],
            color="#F02035",
            weight=6,
            opacity=1.0,
            tooltip=f"Predicted Road Arc: {arc_id}"
        ).add_to(predicted_layer)

        predicted_coordinates.extend([start, end])

        valid_predicted_arcs += 1

    predicted_layer.add_to(m)


    # =================================================
    # 9. OBSERVED GPS TRAJECTORY
    # =================================================

    print("Drawing observed GPS trajectory...")

    gps_layer = folium.FeatureGroup(
        name="Observed GPS Trajectory",
        show=True
    )

    folium.PolyLine(
        locations=gps_coordinates,
        color="#1677E8",
        weight=4,
        opacity=1.0,
        dash_array="8, 10",
        tooltip="Observed GPS Trajectory"
    ).add_to(gps_layer)

    gps_layer.add_to(m)


    # =================================================
    # 10. START AND END MARKERS
    # =================================================

    # GPS starting location

    folium.Marker(
        location=gps_coordinates[0],
        tooltip="GPS Start",
        icon=folium.Icon(
            color="blue",
            icon="play"
        )
    ).add_to(m)

    # GPS ending location

    folium.Marker(
        location=gps_coordinates[-1],
        tooltip="GPS End",
        icon=folium.Icon(
            color="blue",
            icon="stop"
        )
    ).add_to(m)

    # Predicted route start and end

    if predicted_coordinates:

        folium.Marker(
            location=predicted_coordinates[0],
            tooltip="Predicted Route Start",
            icon=folium.Icon(
                color="green",
                icon="play"
            )
        ).add_to(m)

        folium.Marker(
            location=predicted_coordinates[-1],
            tooltip="Predicted Route End",
            icon=folium.Icon(
                color="red",
                icon="flag"
            )
        ).add_to(m)


    # =================================================
    # 11. PROFESSIONAL MAP LEGEND
    # =================================================

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 260px;
        background: white;
        border-radius: 10px;
        padding: 15px;
        z-index: 9999;
        box-shadow: 0 3px 12px rgba(0,0,0,0.25);
        font-family: Arial;
        font-size: 13px;
    ">

        <div style="
            font-size:17px;
            font-weight:bold;
            margin-bottom:10px;
        ">
            PathSnap Map Matching
        </div>

        <hr>

        <div style="margin:10px 0;">
            <span style="
                display:inline-block;
                width:28px;
                border-top:3px solid #E9A23B;
                margin-right:8px;
            "></span>
            Potential Road Candidates
        </div>

        <div style="margin:10px 0;">
            <span style="
                display:inline-block;
                width:28px;
                border-top:5px solid #F02035;
                margin-right:8px;
            "></span>
            Final Predicted Route
        </div>

        <div style="margin:10px 0;">
            <span style="
                display:inline-block;
                width:28px;
                border-top:3px dashed #1677E8;
                margin-right:8px;
            "></span>
            Observed GPS Trajectory
        </div>

        <hr>

        <b>Model Pipeline</b><br>
        XGBoost Emission Scoring<br>
        HMM Transition Scoring<br>
        Viterbi Route Reconstruction

        <hr>

        <b>Trip ID:</b> """ + TRIP_ID + """
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(legend_html)
    )


    # =================================================
    # 12. LAYER CONTROL
    # =================================================

    folium.LayerControl(
        position="topright",
        collapsed=False
    ).add_to(m)


    # =================================================
    # 13. AUTOMATIC ZOOM
    # =================================================

    all_coordinates = (
        gps_coordinates
        + candidate_coordinates
        + predicted_coordinates
    )

    if all_coordinates:

        m.fit_bounds(
            all_coordinates,
            padding=(40, 40)
        )


    # =================================================
    # 14. SAVE HTML MAP
    # =================================================

    m.save(str(OUTPUT_FILE))

    print("\n" + "=" * 55)
    print("PATHSNAP MAP GENERATED SUCCESSFULLY")
    print("=" * 55)

    print("Trip ID:", TRIP_ID)
    print("GPS observations:", len(gps_coordinates))
    print("Potential road arcs:", len(potential_arc_ids))
    print("Predicted road arcs:", valid_predicted_arcs)

    print("\nSaved map:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    generate_map()