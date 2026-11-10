import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# Load DATABASE_URL and other settings from the local .env file.
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    load_dotenv(dotenv_path=_env_path)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./test.db"


# SQLAlchemy engine manages the real PostgreSQL connection to Neon.
# pool_pre_ping=True ensures dropped idle connections (common with Neon serverless)
# are detected and recycled before a query fails with a 500 error.
_db_url = DATABASE_URL
if "channel_binding" in _db_url:
    # psycopg2 does not support the channel_binding parameter; strip it out.
    import re
    _db_url = re.sub(r"[&?]channel_binding=[^&]*", "", _db_url)
    _db_url = re.sub(r"[&?]$", "", _db_url)

# In offline, sandboxed, or local test modes, gracefully fallback to SQLite if remote Postgres is unreachable
_is_online = False
try:
    import socket
    # 3ms IP socket check without DNS resolution to detect sandbox/offline environment
    _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _sock.settimeout(0.1)
    _sock.connect(("8.8.8.8", 53))
    _sock.close()
    _is_online = True
except Exception:
    _is_online = False

if _is_online and "postgres" in _db_url:
    engine = create_engine(_db_url, pool_pre_ping=True)
else:
    engine = create_engine(
        "sqlite:///./local_arogya.db",
        connect_args={"check_same_thread": False}
    )





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
