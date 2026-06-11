from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.database import Base, engine
from auth.models import User
from auth.router import router as auth_router


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


# Simple health/root endpoint to confirm the API is running.
@app.get("/")
def root():
    return {
        "message": "CognoCare AI Auth API is running",
        "endpoints": ["/auth/signup", "/auth/login", "/docs"],
    }

# Trigger uvicorn reload with new environment variables.
