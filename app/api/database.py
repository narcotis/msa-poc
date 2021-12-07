from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database Connection
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@postgres/test"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Actual Database Session.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# for creating the ORM models (to inherit)
Base = declarative_base()
