from typing import List, Optional
from pydantic import BaseModel

# schemas for reading data, and returning data (kind of serializer)


class ValueBase(BaseModel):
    values: str

class ValueCreate(ValueBase):
    pass


class Value(ValueBase):
    id: int

    class Config:
        orm_mode = True


class KeyBase(BaseModel):
    keys: str

class KeyCreate(KeyBase):
    pass

class Key(KeyBase):
    id: int
    values_id: int

    class Config:
        orm_mode = True
