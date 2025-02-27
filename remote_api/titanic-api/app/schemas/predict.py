from typing import Any, List, Optional

from pydantic import BaseModel
from classification_model.processing.validation import TitanicDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "pclass": 1,
                        "sex": 'female',
                        "age": 40,
                        "sibsp": 0,
                        "parch": 0,
                        "fare": 153.4625,
                        "cabin": 'C125',
                        "embarked": 'S',
                        "title": 'Miss'
                        #"cabin_M": 0,
                        #"cabin_C": 1,
                        #"cabin_B": 0,
                        #"cabin_A": 0,
                        #"cabin_G": 0,
                        #"cabin_E": 0,
                        #"cabin_D": 0,
                        #"cabin_F": 0,
                        #"embarked_C": 0,
                        #"embarked_S": 1,
                        #"embarked_Q": 0,
                        #"title_Mrs": 0,
                        #"title_Mr": 0,
                        #"title_Miss": 1,
                        #"title_Other": 0
                    }
                ]
            }
        }
