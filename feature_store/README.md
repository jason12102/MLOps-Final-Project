# Feature Store Module (Feast)

This module adds a local Feast feature store for the diabetes dataset used in this repo.

- `feature_repo/`: Feast repository (`feature_store.yaml`, feature definitions, local registry/online store paths)
- `scripts/`: helper scripts for preparing offline data and reading online features
- `requirements.txt`: module dependencies

## What this module does

1. Builds a Feast-compatible offline source (`.parquet`) from the engineered diabetes dataset.
2. Registers entities, feature views, and feature services with Feast.
3. Materializes features into a local SQLite online store.
4. Retrieves online features for inference-style lookups.

## Prerequisites

- Python 3.10+ recommended
- Diabetes dataset prepared in one of these locations:
  - `data_versioning/outputs/diabetes_v3_features.csv` (preferred)
  - `data_versioning/diabetes.csv`

If needed, generate all data versions first:

```bash
python data_versioning/scripts/build_all.py
```

## Setup

Install dependencies:

```bash
pip install -r feature_store/requirements.txt
```

Prepare the offline feature file for Feast:

```bash
python feature_store/scripts/prepare_offline_data.py
```

Apply feature definitions:

```bash
feast -c feature_store/feature_repo apply
```

Materialize all historical rows into the online store:

```bash
feast -c feature_store/feature_repo materialize-incremental $(python -c "from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat())")
```

PowerShell alternative for timestamp:

```powershell
$ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
feast -c feature_store/feature_repo materialize-incremental $ts
```

## Read online features

```bash
python feature_store/scripts/get_online_features.py --patient-ids 1 2 3
```

## Run integration test

```bash
pytest feature_store/tests/test_prepare_offline_data_integration.py
```

## Key files

- `feature_store/feature_repo/diabetes_features.py`: Feast entities, data source, feature view, feature service
- `feature_store/scripts/prepare_offline_data.py`: builds `feature_repo/data/diabetes_features.parquet`
- `feature_store/scripts/get_online_features.py`: online lookup example
