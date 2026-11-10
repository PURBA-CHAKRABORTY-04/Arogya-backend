"""
auth/dependencies.py
--------------------
FastAPI dependency functions for authentication and role-based access control.

Kept in a separate module (not utils.py) to avoid circular imports:
  utils.py imports from database.py
  routers import from dependencies.py
  dependencies.py imports from both utils.py and models.py
"""

import logging

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .utils import decode_access_token, oauth2_scheme, optional_oauth2_scheme

logger = logging.getLogger(__name__)


def get_optional_current_user(
    token: str | None = Depends(optional_oauth2_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    """Optional JWT validation dependency. Returns User if valid token is provided, else None."""
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        email: str | None = payload.get("sub")
        if not email:
            return None
        return db.query(User).filter(User.email == email).first()
    except Exception:
        return None



def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency — validates the JWT bearer token and returns the
    authenticated User ORM object.

    Raises HTTP 401 if:
      - No token is provided
      - The token signature is invalid
      - The token has expired
      - The user referenced by the token no longer exists in the DB

    Usage in a router:
        @router.get("/some-endpoint")
        def my_endpoint(current_user: User = Depends(get_current_user)):
            ...
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # decode_access_token already raises 401 on signature / expiry failure
    payload = decode_access_token(token)

    email: str | None = payload.get("sub")
    if not email:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if not user:
        logger.warning("Token references non-existent user email hash")
        raise credentials_exception

    return user


def require_role(role: str):
    """
    Dependency factory — wraps get_current_user and additionally enforces
    that the authenticated user has the expected role.

    Raises HTTP 403 if current_user.role != role.

    Usage:
        @router.post("/doctor-only")
        def doctor_endpoint(
            current_user: User = Depends(require_role("doctor")),
        ):
            ...
    """
    def _check_role(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This endpoint requires the '{role}' role.",
            )
        return current_user

    return _check_role


# Convenience shorthand — most patient endpoints use this instead of require_role("patient")
require_patient = require_role("patient")
require_doctor = require_role("doctor")
