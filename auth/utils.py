import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from jose import jwt
from passlib.context import CryptContext


# Load auth settings from .env, with development defaults as fallback.
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-dev-secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing context. Bcrypt stores passwords safely as one-way hashes.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Convert a plain password into a secure bcrypt hash before saving it.
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# Check whether a plain login password matches the stored password hash.
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# Create a signed JWT access token that the frontend can store after login.
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
