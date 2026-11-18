
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from backend.app.services.prediction_service import predict_trip


app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching using XGBoost and HMM",
    version="1.0.0",
)


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    trip_id: str = Field(
        ...,
        min_length=1,
        description="Trip ID to map-match",
    )


@app.get("/")
def home():
    return {
        "application": "PathSnap API",
        "status": "healthy",
        "version": "1.0.0",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PathSnap Prediction API",
        "model": "XGBoost + HMM",
        "ready": True,
    }


@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        result = predict_trip(request.trip_id)

        return {
            "success": True,
            "message": "Prediction completed successfully",
            "data": result,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check backend logs.",
        ) from error
