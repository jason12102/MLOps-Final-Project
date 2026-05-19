"""v1-raw -> v2-cleaned: replace zero-as-missing in 5 columns with column median."""
from pathlib import Path
import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "diabetes.csv"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def main() -> None:
    df = pd.read_csv(CSV)
    for col in ZERO_AS_MISSING:
        median = df.loc[df[col] != 0, col].median()
        df.loc[df[col] == 0, col] = median
    df.to_csv(CSV, index=False)
    print(f"cleaned {CSV} — imputed columns: {ZERO_AS_MISSING}")


if __name__ == "__main__":
    main()
