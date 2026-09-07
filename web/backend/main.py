from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="AquaShield API",
    description="Sonar object detection, anomaly, and triage backend for SIH26057.",
    version="0.3.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)