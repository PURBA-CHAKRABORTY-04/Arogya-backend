import traceback
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .schemas import TokenResponse, UserLogin, UserResponse, UserSignup
from .utils import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, hash_password, verify_password


# All routes in this file start with /auth and appear under the Auth tag in /docs.
router = APIRouter(prefix="/auth", tags=["Auth"])


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

    # Build the ORM object that will become a row in the users table.
    user = User(
        email=payload.email,
        full_name=payload.full_name.strip(),
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


# Verify user credentials and return a JWT token plus user profile.
@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    # Find the account by email and compare the submitted password to the hash.
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
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
