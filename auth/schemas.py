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

    # Required so FastAPI can serialize nested ORM objects without 422 errors.
    model_config = ConfigDict(from_attributes=True)


# ── Forgot-password flow schemas ──────────────────────────────────────────────

# Step 1 — POST /auth/forgot-password: user submits their email.
class ForgotPasswordRequest(BaseModel):
    email: EmailStr


# Step 2 — POST /auth/verify-otp: user submits the 6-digit code they received.
class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


# Step 3 — POST /auth/reset-password: user submits a new password after OTP verification.
class ResetPasswordRequest(BaseModel):
    email: EmailStr
    new_password: str = Field(..., min_length=6)
