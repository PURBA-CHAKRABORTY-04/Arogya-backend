from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.database import Base, engine
from auth.models import User                         # noqa: F401
from auth.router import router as auth_router
from patient.models import PatientInfo               # noqa: F401 — ensures table is created
from patient.router import router as patient_router


# Create the FastAPI application object that Uvicorn runs.
app = FastAPI(title="CognoCare AI Auth API")

# Allow the local Vite frontend to call this backend from the browser.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://cognocareai.pages.dev",
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
