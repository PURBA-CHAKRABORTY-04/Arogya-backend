import os
import csv
import re
import uuid
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from dotenv import load_dotenv

from auth.database import Base
from doctor_matching.models import DoctorProfile

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing. Add it to your .env file.")

# Strip channel_binding if present
if "channel_binding" in DATABASE_URL:
    DATABASE_URL = re.sub(r"[&?]channel_binding=[^&]*", "", DATABASE_URL)
    DATABASE_URL = re.sub(r"[&?]$", "", DATABASE_URL)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Locate dataset file
DATASET_PATH = os.path.join(os.path.dirname(__file__), "Doc_database", "Dataset Doctors - Sheet1.csv")
if not os.path.exists(DATASET_PATH):
    # Fallback search if path is modified
    for fallback_dir in ["Doc_database", "data", "."]:
        cand = os.path.join(os.path.dirname(__file__), fallback_dir, "Dataset Doctors - Sheet1.csv")
        if os.path.exists(cand):
            DATASET_PATH = cand
            break


KOLKATA_KEYWORDS = {
    "kolkata", "calcutta", "kolkata.", "salt lake", "new town", "howrah",
    "dum dum", "dumdum", "behala", "ballygunge", "jadavpur", "tollygunge",
    "alipore", "park street", "esplanade", "shyambazar", "ultadanga"
}


def extract_city(location_str: str) -> str | None:
    """Return 'Kolkata' if the location belongs to Kolkata, else None."""
    if not location_str:
        return None
    lower = location_str.lower()
    for keyword in KOLKATA_KEYWORDS:
        if keyword in lower:
            return "Kolkata"
    return None


def seed_doctors():
    db = SessionLocal()
    try:
        print(f"Reading dataset from: {DATASET_PATH}")
        if not os.path.exists(DATASET_PATH):
            print(f"Error: Dataset file not found at {DATASET_PATH}")
            return

        doctors_to_insert = []

        with open(DATASET_PATH, mode="r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                name = row.get("Name of Doctor", "").strip()
                specialization = row.get("Specialization", "").strip()
                location = row.get("Location/Hospital", "").strip()
                qualification = row.get("Educational Qualification", "").strip()

                if not name or not specialization:
                    continue

                city = extract_city(location)
                if city is None:
                    # Skip doctors not based in Kolkata
                    continue
                email = f"doc_{idx}@arogya.ai"

                # Derive realistic fee based on qualification & role
                fee = 800.0 if any(k in qualification for k in ["DM", "FRCS", "M.Ch", "PhD"]) else 500.0
                if "Director" in specialization or "Senior" in specialization or "Head" in specialization:
                    fee = 1200.0

                doctors_to_insert.append(
                    DoctorProfile(
                        id=uuid.uuid4(),
                        email=email,
                        name=name,
                        role=specialization,
                        qualification=qualification or "MBBS",
                        experience_years=12.0,
                        city=city,
                        consultation_fee=fee,
                        photo_url="https://res.cloudinary.com/demo/image/upload/w_200,h_200,c_thumb,g_face/sample.jpg",
                        description=f"Specialist at {location}. Qualification: {qualification}",
                        verified=True
                    )
                )

        print(f"Parsed {len(doctors_to_insert)} doctor profiles from dataset.")

        # Re-create table to ensure clean state
        from sqlalchemy import text
        print("Re-creating doctor_profiles table...")
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS doctor_profiles CASCADE"))

        Base.metadata.create_all(bind=engine)

        print("Bulk inserting doctor profiles into PostgreSQL...")
        db.bulk_save_objects(doctors_to_insert)
        db.commit()

        count = db.query(DoctorProfile).count()
        kolkata_count = db.query(DoctorProfile).filter(DoctorProfile.city == "Kolkata").count()
        print(f"Doctor seeding completed! Total doctors: {count} (Kolkata: {kolkata_count})")

    except Exception as e:
        db.rollback()
        print(f"Error during seeding: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_doctors()
