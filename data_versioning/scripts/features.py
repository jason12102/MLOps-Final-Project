"""v2-cleaned -> v3-features: add BMI_category, Age_group, Glucose_BMI."""
from pathlib import Path
import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "diabetes.csv"

BMI_BINS = [-float("inf"), 18.5, 25, 30, float("inf")]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]

AGE_BINS = [-float("inf"), 29, 39, 49, 59, float("inf")]
AGE_LABELS = ["20s", "30s", "40s", "50s", "60+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI_category"] = pd.cut(df["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False)
    df["Age_group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    return df


def main() -> None:
    df = add_features(pd.read_csv(CSV))
    df.to_csv(CSV, index=False)
    print(f"engineered features written to {CSV}: BMI_category, Age_group, Glucose_BMI")


if __name__ == "__main__":
    main()
