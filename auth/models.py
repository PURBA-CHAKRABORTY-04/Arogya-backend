import uuid

# pyrefly: ignore [missing-import]
from sqlalchemy import Boolean, Column, DateTime, String, func
# pyrefly: ignore [missing-import]
from sqlalchemy.dialects.postgresql import UUID

from .database import Base


# SQLAlchemy ORM model mapped to the Neon public.users table.
class User(Base):
    __tablename__ = "users"

    # User account fields stored in PostgreSQL.
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(320), unique=True, index=True, nullable=False)
    phone_number = Column(String(20), unique=True, index=True, nullable=True)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="patient")

    # Audit/token timestamp fields that match the existing Neon table.
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    reset_token_expiry = Column(DateTime(timezone=True), nullable=True)
    token_expiry = Column(DateTime(timezone=True), nullable=True)

    # OTP fields for the forgot-password / email-verification flow.
    # otp_code stores a 6-digit code; otp_expires_at marks when it lapses;
    # otp_verified flips to True once the user enters the correct code.
    otp_code = Column(String(6), nullable=True)
    otp_expires_at = Column(DateTime(timezone=True), nullable=True)
    otp_verified = Column(Boolean, nullable=False, default=False)
