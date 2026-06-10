from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# Restrict account roles to the two roles supported by the app UI.
UserRole = Literal["patient", "doctor"]


# Request body expected by POST /auth/signup.
class UserSignup(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=6)
    role: UserRole = "patient"


# Request body expected by POST /auth/login.
class UserLogin(BaseModel):
    email: EmailStr
    password: str
    role: UserRole | None = None


# Public user data returned to the frontend without the password hash.
class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    role: str

    # Allows Pydantic to read values directly from SQLAlchemy User objects.
    model_config = ConfigDict(from_attributes=True)


# Login response containing the JWT plus the logged-in user's public profile.
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
