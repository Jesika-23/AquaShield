"""
backend/main.py
AquaShield backend -- FastAPI entrypoint.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import anomaly, detect
from backend.schemas import CLASS_ID_TO_NAME, HealthResponse
from backend.services.inference_service import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(
    title="AquaShield API",
    description="Sonar object detection, anomaly, and triage backend for SIH26057.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(anomaly.router)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "aquashield-backend"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    model = get_model()
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        classes=list(CLASS_ID_TO_NAME.values()),
    )
