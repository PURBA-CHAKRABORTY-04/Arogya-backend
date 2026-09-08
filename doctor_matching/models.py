import uuid
from sqlalchemy import Column, DateTime, Float, String, Text, func, Boolean
from sqlalchemy.dialects.postgresql import UUID
from auth.database import Base


class DoctorProfile(Base):
    """
    ORM model mapped to the doctor_profiles table in PostgreSQL.
    Stores professional qualifications, location coordinates/address, consultation fee,
    insurance acceptance, availability, and clinical description.
    """
    __tablename__ = "doctor_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(320), unique=True, index=True, nullable=True)
    name = Column(Text, nullable=False)
    role = Column(Text, nullable=False)           # specialty, e.g. "Integrative Neurologist"
    qualification = Column(Text, nullable=False)  # e.g. "MBBS, MD, DM (Neurology)"
    experience_years = Column(Float, nullable=True)
    city = Column(String(255), nullable=False)
    address = Column(Text, nullable=True)         # Full address string
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    consultation_fee = Column(Float, nullable=False, default=0.0)
    photo_url = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    verified = Column(Boolean, default=True, nullable=False)
    is_available = Column(Boolean, default=True, nullable=False)
    insurance_accepted = Column(Text, nullable=True) # JSON or comma-separated list of insurance providers
    average_rating = Column(Float, default=4.5, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class DoctorRankingInteraction(Base):
    """
    ORM model mapped to the doctor_ranking_interactions table in PostgreSQL.
    Logs impressions, clicks, and bookings from doctor search results to enable offline
    calibration and tuning of formula weights against real user behavior.
    """
    __tablename__ = "doctor_ranking_interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), nullable=False, index=True)
    patient_email = Column(String(320), nullable=True, index=True)
    search_query = Column(Text, nullable=True)
    doctor_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    rank_position = Column(Float, nullable=True)
    raw_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True)
    score_breakdown = Column(Text, nullable=True)  # JSON-encoded score components
    interaction_type = Column(String(50), nullable=False)  # 'impression' | 'click' | 'booking'

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

