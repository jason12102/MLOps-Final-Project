from pathlib import Path

import pandas as pd

BMI_BINS = [-float("inf"), 18.5, 25, 30, float("inf")]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]
AGE_BINS = [-float("inf"), 29, 39, 49, 59, float("inf")]
AGE_LABELS = ["20s", "30s", "40s", "50s", "60+"]


def pick_input_csv(project_root: Path) -> Path:
    candidates = [
        project_root / "data_versioning" / "outputs" / "diabetes_v3_features.csv",
        project_root / "data_versioning" / "diabetes.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No diabetes CSV found. Expected one of:\n"
        f"- {candidates[0]}\n"
        f"- {candidates[1]}"
    )


def ensure_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "BMI_category" not in out.columns:
        out["BMI_category"] = pd.cut(
            out["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False
        )
    if "Age_group" not in out.columns:
        out["Age_group"] = pd.cut(out["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    if "Glucose_BMI" not in out.columns:
        out["Glucose_BMI"] = out["Glucose"] * out["BMI"]
    return out


def build_feast_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_engineered_columns(df)
    out = out.reset_index(drop=True)
    out["patient_id"] = out.index + 1

    base_ts = pd.Timestamp.utcnow().floor("s") - pd.Timedelta(days=30)
    out["event_timestamp"] = base_ts + pd.to_timedelta(out.index, unit="m")

    int_cols = ["Pregnancies", "Age", "patient_id"]
    float_cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Glucose_BMI",
    ]

    for col in int_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int64")
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    out["BMI_category"] = out["BMI_category"].astype("string").fillna("Unknown")
    out["Age_group"] = out["Age_group"].astype("string").fillna("Unknown")

    ordered_columns = [
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
    return out[ordered_columns]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    feature_repo_dir = project_root / "feature_store" / "feature_repo"
    output_path = feature_repo_dir / "data" / "diabetes_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_csv = pick_input_csv(project_root)
    raw_df = pd.read_csv(source_csv)
    feast_df = build_feast_frame(raw_df)
    feast_df.to_parquet(output_path, index=False)

    print(f"source csv: {source_csv}")
    print(f"output parquet: {output_path}")
    print(f"rows: {len(feast_df)}, columns: {len(feast_df.columns)}")


if __name__ == "__main__":
    main()
