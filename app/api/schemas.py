from typing import List, Optional
from pydantic import BaseModel

# schemas for reading data, and returning data (kind of serializer)
# type hint를 통해 parsing을 도와줌, validation check는 아님

class ValueBase(BaseModel):
    value: str

# Create부분은 read할때 반환되지 않음
class ValueCreate(ValueBase):
    pass


class Value(ValueBase):
    id: int

    class Config:
        orm_mode = True


class KeyBase(BaseModel):
    key: str

class KeyCreate(KeyBase):
    pass

class Key(KeyBase):
    id: int
    value_id: int

    # ORM mode를 이용하면 dict처럼이 아니라 ORM처럼 사용 가능
    # id = data["id"] / id = data.id
    class Config:
        orm_mode = True
