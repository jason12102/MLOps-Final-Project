from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "feature_store" / "scripts" / "prepare_offline_data.py"


def load_prepare_module():
    spec = spec_from_file_location("prepare_offline_data", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_feast_frame_schema_and_values():
    prep = load_prepare_module()

    try:
        source_csv = prep.pick_input_csv(PROJECT_ROOT)
    except FileNotFoundError:
        pytest.skip("diabetes source CSV not found in data_versioning/")

    raw_df = pd.read_csv(source_csv)
    feast_df = prep.build_feast_frame(raw_df)

    expected_columns = [
        "patient_id",
        "event_timestamp",
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "BMI_category",
        "Age_group",
        "Glucose_BMI",
    ]

    assert list(feast_df.columns) == expected_columns
    assert len(feast_df) == len(raw_df)
    assert feast_df["patient_id"].iloc[0] == 1
    assert feast_df["patient_id"].is_monotonic_increasing
    assert feast_df["patient_id"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(feast_df["event_timestamp"])
    assert feast_df["BMI_category"].notna().all()
    assert feast_df["Age_group"].notna().all()

    expected_glucose_bmi = feast_df["Glucose"] * feast_df["BMI"]
    assert np.allclose(feast_df["Glucose_BMI"], expected_glucose_bmi, equal_nan=True)


def test_prepare_main_writes_parquet():
    prep = load_prepare_module()

    try:
        prep.pick_input_csv(PROJECT_ROOT)
    except FileNotFoundError:
        pytest.skip("diabetes source CSV not found in data_versioning/")

    prep.main()

    output_path = (
        PROJECT_ROOT
        / "feature_store"
        / "feature_repo"
        / "data"
        / "diabetes_features.parquet"
    )
    assert output_path.exists()

    df = pd.read_parquet(output_path)
    assert len(df) > 0
    assert "patient_id" in df.columns
    assert "event_timestamp" in df.columns
