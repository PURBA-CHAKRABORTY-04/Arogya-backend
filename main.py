import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.database import Base, engine
from auth.models import User                         # noqa: F401
from auth.router import router as auth_router
from dashboard.patient.models import PatientInfo               # noqa: F401 — ensures table is created
from dashboard.patient.timeline_models import TimelineEntry    # noqa: F401 — ensures timeline table is created
from symptom_triage.models import TriageSession      # noqa: F401 — ensures triage session table is created
from dashboard.patient.router import router as patient_router
from symptom_triage.router import router as symptom_triage_router
from complaints.models import Complaint                # noqa: F401 — ensures complaint table is created
from complaints.router import router as complaints_router
from dashboard.upload.router import router as upload_router
from dashboard.biometric.models import BiometricMetric          # noqa: F401 — ensures biometric table is created
from dashboard.biometric.router import router as biometric_router
from dashboard.schedule.models import ScheduleEvent              # noqa: F401 — ensures schedule_events table is created
from dashboard.schedule.router import router as schedule_router
from doctor_matching.models import DoctorProfile              # noqa: F401 — ensures doctor_profiles table is created
from doctor_matching.router import router as doctor_matching_router
from dashboard.voice.router import router as voice_router

# ── Issue #11: Structured logging configuration ───────────────────────────────
# Replace all bare print() with Python's logging module.
# LOG_LEVEL can be overridden via environment variable (e.g. LOG_LEVEL=DEBUG).
import os
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# Create the FastAPI application object that Uvicorn runs.
app = FastAPI(title="CognoCare AI Auth API")

# Allow the local Vite frontend to call this backend from the browser.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://arogya.pages.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables from SQLAlchemy models if they do not already exist.
Base.metadata.create_all(bind=engine)

# Register all /auth routes, such as /auth/signup and /auth/login.
app.include_router(auth_router)

# Register all /patient routes for patient profile CRUD.
app.include_router(patient_router)

# Register /symptom_triage routes for Gemini LLM Triage
app.include_router(symptom_triage_router)

# Register /complaints routes for Patient Complaints
app.include_router(complaints_router, prefix="/complaints")

# Register /upload routes for Cloudinary file uploads (PDF / JPG / PNG)
app.include_router(upload_router)

# Register /biometric routes for patient biometric metric CRUD
app.include_router(biometric_router)

# Register /schedule routes for patient schedule events CRUD
app.include_router(schedule_router)

# Register /doctor-matching routes for patient matching
app.include_router(doctor_matching_router)

# Register /voice routes for server-side Gemini audio transcription (Path B)
app.include_router(voice_router)

# Simple health/root endpoint to confirm the API is running.
@app.get("/")
def root():
    return {
        "message": "CognoCare AI Auth API is running",
        "endpoints": [
            "/auth/signup",
            "/auth/login",
            "/patient/patient-info",
            "/docs",
        ],
    }

# Trigger uvicorn reload with new environment variables.
# Reload triggered after installing openai and other packages.
