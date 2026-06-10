import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# Load DATABASE_URL and other settings from the local .env file.
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing. Add it to your .env file.")

# SQLAlchemy engine manages the real PostgreSQL connection to Neon.
engine = create_engine(DATABASE_URL)

# SessionLocal creates one database session per request.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base is inherited by all ORM models, such as the User model.
Base = declarative_base()


# FastAPI dependency that opens a DB session and always closes it afterward.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
