import os
from contextlib import asynccontextmanager
from pathlib import Path

import h2o
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    "/app/model/GBM_1_AutoML_3_20260525_194002"
))

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    h2o.init(nthreads=-1, max_mem_size="2G")
    h2o.no_progress()
    
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    
    model = h2o.load_model(str(MODEL_PATH))
    print(f"Loaded model: {model.model_id}")
    
    yield
    
    h2o.cluster().shutdown()


app = FastAPI(
    title="Diabetes Prediction API",
    description="FastAPI service serving H2O AutoML GBM model for diabetes prediction",
    version="1.0.0",
    lifespan=lifespan,
)


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    Glucose: float = Field(..., gt=0, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=0, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., gt=0, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Diabetes pedigree function")
    Age: int = Field(..., ge=1, description="Age in years")
    BMI_category: str = Field(..., description="BMI category (Underweight/Normal/Overweight/Obese)")
    Age_group: str = Field(..., description="Age group (Young/Middle-aged/Senior)")
    Glucose_BMI: float = Field(..., description="Glucose * BMI interaction feature")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pregnancies": 6,
                    "Glucose": 148.0,
                    "BloodPressure": 72.0,
                    "SkinThickness": 35.0,
                    "Insulin": 0.0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50,
                    "BMI_category": "Obese",
                    "Age_group": "Middle-aged",
                    "Glucose_BMI": 4972.8
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted outcome (0=No Diabetes, 1=Diabetes)")
    probability_no_diabetes: float = Field(..., description="Probability of no diabetes")
    probability_diabetes: float = Field(..., description="Probability of diabetes")
    model_id: str = Field(..., description="ID of the model used")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str | None


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None,
        model_id=model.model_id if model else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: DiabetesInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_dict = input_data.model_dump()
    df = pd.DataFrame([input_dict])
    
    hf = h2o.H2OFrame(df)
    for col in ["BMI_category", "Age_group"]:
        hf[col] = hf[col].asfactor()
    
    predictions = model.predict(hf)
    pred_df = predictions.as_data_frame()
    
    prediction = int(pred_df["predict"].iloc[0])
    prob_0 = float(pred_df["p0"].iloc[0])
    prob_1 = float(pred_df["p1"].iloc[0])
    
    return PredictionResponse(
        prediction=prediction,
        probability_no_diabetes=round(prob_0, 4),
        probability_diabetes=round(prob_1, 4),
        model_id=model.model_id,
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(inputs: list[DiabetesInput]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(inputs) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limited to 1000")
    
    records = [inp.model_dump() for inp in inputs]
    df = pd.DataFrame(records)
    
    hf = h2o.H2OFrame(df)
    for col in ["BMI_category", "Age_group"]:
        hf[col] = hf[col].asfactor()
    
    predictions = model.predict(hf)
    pred_df = predictions.as_data_frame()
    
    results = []
    for i in range(len(pred_df)):
        results.append({
            "prediction": int(pred_df["predict"].iloc[i]),
            "probability_no_diabetes": round(float(pred_df["p0"].iloc[i]), 4),
            "probability_diabetes": round(float(pred_df["p1"].iloc[i]), 4),
        })
    
    return {"predictions": results, "model_id": model.model_id}
