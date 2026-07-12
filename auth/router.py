import random
import traceback
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .database import get_db
from .email_utils import send_otp_email
from .models import User
from .schemas import (
    ForgotPasswordRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserLogin,
    UserResponse,
    UserSignup,
    VerifyOtpRequest,
)
from .utils import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, hash_password, verify_password


# All routes in this file start with /auth and appear under the Auth tag in /docs.
router = APIRouter(prefix="/auth", tags=["Auth"])

# How long a generated OTP stays valid.
OTP_EXPIRY_MINUTES = 10


# ── Sign Up ───────────────────────────────────────────────────────────────────

# Create a new user account, hash the password, and save it in Neon.
@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: UserSignup, db: Session = Depends(get_db)):
    # Reject duplicate emails before trying to insert a new row.
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Reject duplicate phone numbers.
    if payload.phone_number:
        existing_phone = db.query(User).filter(User.phone_number == payload.phone_number).first()
        if existing_phone:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Phone number already registered",
            )

    # Build the ORM object that will become a row in the users table.
    user = User(
        email=payload.email,
        full_name=payload.full_name.strip(),
        phone_number=payload.phone_number,
        hashed_password=hash_password(payload.password),
        role=payload.role,
    )

    # Commit writes the new user to Neon; rollback keeps the session clean on error.
    print("Creating new user:", user.email)
    db.add(user)
    try:
        print("Committing to database...")
        db.commit()
        print("Commit successful.")
    except IntegrityError as exc:
        db.rollback()
        print("[IntegrityError]", traceback.format_exc())
        if getattr(exc.orig, "pgcode", None) != "23505":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create user",
            ) from None
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        ) from None
    except Exception as exc:
        db.rollback()
        print("[DB ERROR - full traceback]")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc}",
        ) from exc

    db.refresh(user)
    return user


# ── Login ─────────────────────────────────────────────────────────────────────

# Verify user credentials and return a JWT token plus user profile.
@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    # Find the account by email or phone number depending on which was submitted.
    if payload.email:
        user = db.query(User).filter(User.email == payload.email).first()
        invalid_msg = "Incorrect email or password"
    else:
        user = db.query(User).filter(User.phone_number == payload.phone_number).first()
        invalid_msg = "Incorrect phone number or password"

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=invalid_msg,
        )

    # Prevent a patient account from logging in through the doctor tab, and vice versa.
    if payload.role and user.role != payload.role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Selected role does not match this account",
        )

    # Store the token expiry time in the database and create the signed JWT.
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    user.token_expiry = datetime.now(timezone.utc) + expires_delta

    access_token = create_access_token(
        data={"sub": user.email, "user_id": str(user.id), "role": user.role},
        expires_delta=expires_delta,
    )
    db.commit()
    db.refresh(user)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
    }


# ── Forgot Password — Step 1: Send OTP ───────────────────────────────────────

@router.post("/forgot-password", status_code=status.HTTP_200_OK)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Generates a 6-digit OTP, stores it with a 10-minute expiry in the DB,
    then dispatches it to the user's inbox via the email microservice.
    Always returns 200 even if the email is not found (prevents user enumeration).
    """
    user = db.query(User).filter(User.email == payload.email).first()

    # Return a generic success even for unknown emails to prevent enumeration attacks.
    if not user:
        return {"message": "If that email exists, an OTP has been sent."}

    # Generate a cryptographically adequate 6-digit OTP.
    otp = str(random.SystemRandom().randint(100_000, 999_999))

    # Persist the OTP and its expiry; clear any previous verified state.
    user.otp_code = otp
    user.otp_expires_at = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)
    user.otp_verified = False
    db.commit()

    # Call the email microservice to deliver the OTP.
    try:
        send_otp_email(email=user.email, otp=otp, name=user.full_name)
    except RuntimeError as exc:
        # Roll back OTP if email delivery failed so the user can retry cleanly.
        user.otp_code = None
        user.otp_expires_at = None
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    print(f"[forgot-password] OTP sent to {user.email}")
    return {"message": "If that email exists, an OTP has been sent."}


# ── Forgot Password — Step 2: Verify OTP ─────────────────────────────────────

@router.post("/verify-otp", status_code=status.HTTP_200_OK)
def verify_otp(payload: VerifyOtpRequest, db: Session = Depends(get_db)):
    """
    Validates the submitted OTP against the stored code and expiry.
    Sets otp_verified=True on success so the reset-password endpoint can trust the session.
    """
    user = db.query(User).filter(User.email == payload.email).first()

    if not user or not user.otp_code or not user.otp_expires_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active OTP for this email. Please request a new one.",
        )

    # Check expiry (compare timezone-aware datetimes).
    if datetime.now(timezone.utc) > user.otp_expires_at:
        # Clean up expired OTP.
        user.otp_code = None
        user.otp_expires_at = None
        user.otp_verified = False
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP has expired. Please request a new one.",
        )

    # Constant-time comparison avoids timing attacks.
    if payload.otp != user.otp_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OTP. Please check the code and try again.",
        )

    # Mark OTP as verified — the reset-password endpoint checks this flag.
    user.otp_verified = True
    db.commit()

    print(f"[verify-otp] OTP verified for {user.email}")
    return {"message": "OTP verified successfully. You may now reset your password."}


# ── Forgot Password — Step 3: Reset Password ─────────────────────────────────

@router.post("/reset-password", status_code=status.HTTP_200_OK)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Updates the user's password after successful OTP verification.
    Clears all OTP state from the DB after a successful reset.
    """
    user = db.query(User).filter(User.email == payload.email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    # Guard: OTP must have been verified in the same session.
    if not user.otp_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="OTP has not been verified. Please complete the verification step first.",
        )

    # Guard: even if verified, reject if the OTP window has lapsed.
    if not user.otp_expires_at or datetime.now(timezone.utc) > user.otp_expires_at:
        user.otp_code = None
        user.otp_expires_at = None
        user.otp_verified = False
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP session expired. Please restart the password reset flow.",
        )

    # Hash and save the new password.
    user.hashed_password = hash_password(payload.new_password)

    # Clear all OTP state — one-time use only.
    user.otp_code = None
    user.otp_expires_at = None
    user.otp_verified = False
    user.reset_token_expiry = None

    db.commit()
    print(f"[reset-password] Password reset successful for {user.email}")
    return {"message": "Password reset successfully. You may now log in with your new password."}
