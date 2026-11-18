
const API_BASE_URL = "http://127.0.0.1:8000";

async function predictRoute() {
    const tripId = document.getElementById("tripId").value.trim();
    const status = document.getElementById("status");
    const button = document.getElementById("predictBtn");
    const results = document.getElementById("results");

    if (!tripId) {
        status.textContent = "Please enter a trip ID.";
        status.style.color = "red";
        return;
    }

    button.disabled = true;
    button.textContent = "Predicting...";
    status.textContent = "Connecting to PathSnap API...";
    status.style.color = "#2563eb";
    results.classList.add("hidden");

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                trip_id: tripId
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Prediction failed.");
        }

        if (!result.success || !result.data) {
            throw new Error("Unexpected API response.");
        }

        const data = result.data;

        document.getElementById("gpsCount").textContent =
            data.gps_observations;

        document.getElementById("predictedCount").textContent =
            data.predicted_observations;

        document.getElementById("missingCount").textContent =
            data.missing_predictions;

        document.getElementById("arcCount").textContent =
            data.unique_predicted_arcs;

        document.getElementById("routeLength").textContent =
            `Total predicted route arcs: ${data.reconstructed_route.length}`;

        const route = data.reconstructed_route;

        document.getElementById("routeOutput").textContent =
            route.length > 0
                ? route.join(" → ")
                : "No route could be reconstructed.";

        results.classList.remove("hidden");

        status.textContent = result.message;
        status.style.color = "green";

    } catch (error) {
        status.textContent = error.message;
        status.style.color = "red";

    } finally {
        button.disabled = false;
        button.textContent = "Predict Route";
    }
}
