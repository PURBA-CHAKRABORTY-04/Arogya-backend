from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

from auth.database import Base

class Complaint(Base):
    __tablename__ = "complaints"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    patient_email = Column(String, ForeignKey("users.email"), nullable=False)
    category = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, default="Submitted", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User")
