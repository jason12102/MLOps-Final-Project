# MLOps Final Project

## Overview

This project demonstrates an end-to-end MLOps pipeline composed of four core modules:

- **Data Versioning**
- **Feature Store**
- **AutoML & MLflow**
- **Model Monitoring**

Each module is implemented independently while being orchestrated through a centralized Streamlit application (`app.py`) that serves as an interactive demo dashboard.

The objective of this project is to simulate a production-style machine learning workflow focused on:
- reproducibility
- experiment tracking
- feature management
- model observability
- modular architecture

---

## How to Run

### Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ (for running the demo script locally)

### 1. Start All Services

From the project root directory:

```bash
docker-compose up --build
```

This starts three services:
- **model-serving** - FastAPI model inference API on port 8000
- **monitoring** - Evidently drift detection service
- **monitoring-ui** - Evidently dashboard on port 8001

### 2. Run the Demo

In a new terminal:

```bash
cd demo
pip install -r requirements.txt
python run_demo.py --step all
```

Demo options:
- `--step original` - Run with original test data only
- `--step modified` - Run with modified (drifted) test data only
- `--step all` - Run both and compare results

### 3. View Results

- **Model API**: http://localhost:8000
- **Evidently Monitoring UI**: http://localhost:8001

### Stopping Services

```bash
docker-compose down
```

---

## Project Structure

```text
mlops-final-project/
│
├── app.py
│
├── data_versioning/
│
├── feature_store/
│
├── automl/
│
├── model_serving/
│
├── model_monitoring/
│
├── demo/
│
├── eda/
│
├── docker-compose.yml
│
└── README.md