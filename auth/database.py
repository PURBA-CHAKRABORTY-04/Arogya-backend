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
# pool_pre_ping=True ensures dropped idle connections (common with Neon serverless)
# are detected and recycled before a query fails with a 500 error.
_db_url = DATABASE_URL
if "channel_binding" in _db_url:
    # psycopg2 does not support the channel_binding parameter; strip it out.
    import re
    _db_url = re.sub(r"[&?]channel_binding=[^&]*", "", _db_url)
    _db_url = re.sub(r"[&?]$", "", _db_url)

engine = create_engine(_db_url, pool_pre_ping=True)

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
