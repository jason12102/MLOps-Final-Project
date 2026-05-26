# Model Serving - Diabetes Prediction API

FastAPI service serving the best H2O AutoML GBM model for diabetes prediction.

## Quick Start

### Using Docker Compose (Recommended)
```bash
cd model_serving
docker-compose up --build
```

### Using Docker directly
```bash
cd model_serving
docker build -t diabetes-model-serving .
docker run -p 8000:8000 diabetes-model-serving
```

### Local Development (requires Java 17+)
```bash
cd model_serving
pip install -r requirements.txt
cd app
uvicorn main:app --reload --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions (max 1000) |
| `/docs` | GET | Interactive Swagger UI |

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Response Format

```json
{
  "prediction": 1,
  "probability_no_diabetes": 0.2534,
  "probability_diabetes": 0.7466,
  "model_id": "GBM_1_AutoML_3_20260525_194002"
}
```

## Model Info

- **Algorithm**: Gradient Boosting Machine (GBM)
- **Framework**: H2O AutoML
- **Metrics**: AUC=0.8017, Accuracy=79.12%
- **Data Version**: v3_features (with engineered features)
