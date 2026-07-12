from typing import Literal
from uuid import UUID

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


# Restrict account roles to the two roles supported by the app UI.
UserRole = Literal["patient", "doctor"]


# Request body expected by POST /auth/signup.
class UserSignup(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=255)
    phone_number: str = Field(..., min_length=7, max_length=20, pattern=r"^\+?[\d\s\-\(\)]{7,20}$")
    password: str = Field(..., min_length=6)
    role: UserRole = "patient"


# Request body expected by POST /auth/login.
# Accepts EITHER email OR phone_number (at least one must be provided).
class UserLogin(BaseModel):
    email: EmailStr | None = None
    phone_number: str | None = None
    password: str
    role: UserRole | None = None

    @model_validator(mode="after")
    def require_email_or_phone(self):
        if not self.email and not self.phone_number:
            raise ValueError("Either 'email' or 'phone_number' must be provided.")
        return self


# Public user data returned to the frontend without the password hash.
class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    role: str
    phone_number: str | None = None

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

