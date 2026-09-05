"""
backend/main.py
AquaShield backend -- FastAPI entrypoint.
Task 1: placeholder app to verify the toolchain. Real routes are wired
in Task 2 onward.
"""
from fastapi import FastAPI

app = FastAPI(
    title="AquaShield API",
    description="Sonar object detection, anomaly, and triage backend for SIH26057.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"status": "ok", "service": "aquashield-backend"}
