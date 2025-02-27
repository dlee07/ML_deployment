import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from classification_model import __version__ as model_version
from classification_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/titanic", response_model=schemas.Titanic, status_code=200)
def titanic() -> dict:
    """
    Root Get
    """
    titanic = schemas.Titanic(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return titanic.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleTitanicDataInputs) -> Any:
    """
    Make house price predictions with the TID regression model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # Advanced: You can improve performance of your API by rewriting the
    # `make prediction` function to be async and using await here.
    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results
