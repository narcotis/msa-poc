from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Table
from sqlalchemy.orm import relationship

# SQLAlchemy model 생성 전에 반드시 import
from .database import Base

# N:M relation (association table)
keyvalue = Table('keyvalue', Base.metadata,
                 Column('key_id', Integer, ForeignKey('keys.id')),
                 Column('value_id', Integer, ForeignKey('values.id'))
                 )


class Key(Base):
    # table namespace
    __tablename__ = "keys"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String)
    # value_id = Column(Integer, ForeignKey("values.id"))

    # related name
    value = relationship("Value", secondary=keyvalue, back_populates="key")


class Value(Base):
    # table namespace
    __tablename__ = "values"
    id = Column(Integer, primary_key=True, index=True)
    value = Column(String)

    # 참조 / 역참조
    key = relationship("Key", secondary=keyvalue, back_populates="value")