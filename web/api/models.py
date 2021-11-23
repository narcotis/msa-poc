from .database import Base
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

class Key(Base):
    # table namespace
    __tablename__ = "keys"

    id = Column(Integer, primary_key=True, index=True)
    keys = Column(String)
    values_id = Column(Integer, ForeignKey("values.id"))

    # related name
    value = relationship("Value", back_populates="values")

class Value(Base):
    # table namespace
    __tablename__ = "values"

    id = Column(Integer, primary_key=True, index=True)
    values = Column(String)

    key = relationship("Key", back_populates='keys')