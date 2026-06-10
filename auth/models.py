import uuid

from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID

from .database import Base


# SQLAlchemy ORM model mapped to the Neon public.users table.
class User(Base):
    __tablename__ = "users"

    # User account fields stored in PostgreSQL.
    id = Column(UUID(as_uuid=True), primary_key=False, default=uuid.uuid4)
    email = Column(String(320), primary_key=True, index=True, nullable=False)
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
