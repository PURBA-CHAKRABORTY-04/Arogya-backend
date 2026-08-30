import uuid
from sqlalchemy import Column, DateTime, Float, String, Text, func, Boolean
from sqlalchemy.dialects.postgresql import UUID
from auth.database import Base


class DoctorProfile(Base):
    """
    ORM model mapped to the doctor_profiles table in PostgreSQL.
    Stores professional qualifications, location, consultation fee, and clinical description.
    """
    __tablename__ = "doctor_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(320), unique=True, index=True, nullable=True)
    name = Column(Text, nullable=False)
    role = Column(Text, nullable=False)           # specialty, e.g. "Integrative Neurologist"
    qualification = Column(Text, nullable=False)  # e.g. "MD, DM (Neurology)"
    experience_years = Column(Float, nullable=True)
    city = Column(String(255), nullable=False)
    consultation_fee = Column(Float, nullable=False, default=0.0)
    photo_url = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    verified = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
