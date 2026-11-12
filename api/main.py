
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal


# 1. Load the trained model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

model = joblib.load(MODEL_DIR / "random_forest_model.pkl")
model_features = joblib.load(MODEL_DIR / "model_features.pkl")


# 2. Create the API
app = FastAPI(
    title="FactoryPulse API",
    description="Predictive maintenance failure prediction API",
    version="1.0.0"
)


# 3. Define the input data
class MachineInput(BaseModel):
    type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: int = Field(gt=0)
    torque_nm: float = Field(gt=0)
    tool_wear_min: int = Field(ge=0)


# 4. Create the prediction endpoint
@app.post("/predict")
def predict_failure(data: MachineInput):

    input_df = pd.DataFrame([data.model_dump()])

    # Apply the same encoding used during training
    input_df = pd.get_dummies(
        input_df,
        columns=["type"],
        dtype=int
    )

    # Match the exact training feature columns and order
    input_df = input_df.reindex(
        columns=model_features,
        fill_value=0
    )

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    return {
        "prediction": "Failure" if prediction == 1 else "Normal",
        "failure_probability_pct": round(probability * 100, 2),
        "risk_message": (
            "Potential failure detected"
            if prediction == 1
            else "No failure predicted"
        )
    }


# 5. API health check
@app.get("/")
def home():
    return {
        "message": "FactoryPulse API is running!"
    }
