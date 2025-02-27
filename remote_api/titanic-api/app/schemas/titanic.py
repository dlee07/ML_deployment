from pydantic import BaseModel


class Titanic(BaseModel):
    name: str
    api_version: str
    model_version: str
