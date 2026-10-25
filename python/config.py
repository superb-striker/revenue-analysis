import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Load .env from the project root (one level up from this python/ folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_engine() -> Engine:
    user = os.environ["POSTGRES_USERNAME"]
    password = os.environ["POSTGRES_PASSWORD"]
    database = os.environ["POSTGRES_DATABASE"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url)
