# H2O AutoML Demo — MLOps Final Project

## What this runs
| Service | URL | Description |
|---|---|---|
| Jupyter Lab | http://localhost:8888 | Run the AutoML notebook |
| H2O Flow UI | http://localhost:54321 | Visual H2O dashboard |

## Prerequisites
- Docker Desktop installed and running
- At least 6GB RAM available to Docker

## Quickstart

```bash
# 1. Clone the repo and enter this folder
git clone <repo-url>
cd MLOps-Final-Project/docker-demo

# 2. Build and start
docker compose up --build

# 3. Open your browser
#    Jupyter → http://localhost:8888
#    H2O Flow → http://localhost:54321

# 4. In Jupyter, open notebooks/h2o_automl_diabetes.ipynb and run all cells

# 5. Stop when done
docker compose down
```

## For the presentation
- Open **H2O Flow** in one browser tab to show the live cluster and leaderboard visually
- Open **Jupyter** in another tab to walk through the notebook cells
- Results (leaderboards, saved models) persist in the `results/` folder on your host machine via the volume mount

## Folder structure
```
docker-demo/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── start.sh
├── .dockerignore
├── notebooks/
│   └── h2o_automl_diabetes.ipynb
└── results/          ← created on first run, persists outside container
```
