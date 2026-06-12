# patient/models.py
import uuid

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from auth.database import Base


class PatientInfo(Base):
    """ORM model mapped to the patient_info table in PostgreSQL.

    The email column is a FK to users.email — it is immutable once set.
    age and bmi are NOT stored here; they are computed at response time
    by the PatientInfoResponse Pydantic schema.
    """

    __tablename__ = "patient_info"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # email FK links to users.email — immutable once set
    email = Column(
        String(320),
        ForeignKey("users.email", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    # Core demographics — required at creation
    name           = Column(String(255), nullable=False)
    dob            = Column(Date, nullable=False)            # uneditable after creation
    contact_number = Column(String(20), nullable=False)      # stored as text (supports + prefix)
    gender         = Column(Text, nullable=False)
    city           = Column(String(100), nullable=False)

    # Immutable medical identifiers
    blood_group = Column(String(5), nullable=True)           # uneditable after creation

    # Physical metrics — editable
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)

    # Medical history arrays — editable
    chronic_conditions = Column(ARRAY(Text), nullable=True, default=list)
    known_allergies    = Column(ARRAY(Text), nullable=True, default=list)
    active_medications = Column(ARRAY(Text), nullable=True, default=list)

    # Lifestyle fields — editable
    smoking_status      = Column(String(20), nullable=True)  # "Non-Smoker" | "Tobacco User"
    alcohol_consumption = Column(String(20), nullable=True)  # "Alcohol-Free" | "Consumes Alcohol"

    # Emergency contact — editable
    emergency_contact_name         = Column(String(255), nullable=True)
    emergency_contact_phone        = Column(String(20),  nullable=True)
    emergency_contact_relationship = Column(String(100), nullable=True)

    # Audit timestamps — managed by the DB
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
