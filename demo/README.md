# Demo: Model Validation and Drift Detection

This demo shows how to validate model performance and detect data drift.

## Quick Start

### 1. Start the services
```bash
# From project root
docker-compose up --build
```

### 2. Run the demo (in a new terminal)
```bash
cd demo
pip install -r requirements.txt
python run_demo.py --step all
```

## What the Demo Does

### Step 1: Original Data
- Loads 20% of diabetes dataset as test data
- Sends to model API for predictions
- Calculates metrics (AUC, Accuracy, F1, etc.)

### Step 2: Modified Data
Creates "drifted" data by:
1. **Swapping Glucose <-> BloodPressure** - Simulates sensor misconfiguration
2. **Doubling Insulin values** - Simulates measurement drift

### Step 3: Comparison
- Shows metric degradation between original and modified
- Saves data for Evidently monitoring

## Viewing Drift in Evidently UI

1. After running the demo, open: http://localhost:8001
2. View the drift report showing:
   - Feature distribution changes
   - Statistical drift detection
   - Model performance comparison

## Demo Commands

```bash
# Run everything
python run_demo.py --step all

# Run only original data evaluation
python run_demo.py --step original

# Run only modified data evaluation  
python run_demo.py --step modified
```

## Expected Results

| Metric    | Original | Modified | Status   |
|-----------|----------|----------|----------|
| Accuracy  | ~0.78    | ~0.55    | DEGRADED |
| AUC       | ~0.82    | ~0.50    | DEGRADED |
| F1        | ~0.65    | ~0.45    | DEGRADED |

The significant performance drop demonstrates why drift detection is critical!
