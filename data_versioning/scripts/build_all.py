"""One-shot: from raw diabetes.csv, write all 3 versions to outputs/."""
from pathlib import Path
import shutil
import pandas as pd

from clean import clean_df, ZERO_AS_MISSING
from features import add_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "diabetes.csv"
OUT = ROOT / "outputs"


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"missing raw input: {RAW} (drop the Kaggle file there first)")

    df_raw = pd.read_csv(RAW)
    if not any((df_raw[c] == 0).any() for c in ZERO_AS_MISSING):
        print(f"warning: no zeros found in {ZERO_AS_MISSING} — {RAW} may not be the raw v1 file")

    OUT.mkdir(exist_ok=True)

    v1 = OUT / "diabetes_v1_raw.csv"
    v2 = OUT / "diabetes_v2_cleaned.csv"
    v3 = OUT / "diabetes_v3_features.csv"

    shutil.copyfile(RAW, v1)
    print(f"wrote {v1} ({df_raw.shape[0]}x{df_raw.shape[1]})")

    df_cleaned = clean_df(df_raw)
    df_cleaned.to_csv(v2, index=False)
    print(f"wrote {v2} ({df_cleaned.shape[0]}x{df_cleaned.shape[1]})")

    df_features = add_features(df_cleaned)
    df_features.to_csv(v3, index=False)
    print(f"wrote {v3} ({df_features.shape[0]}x{df_features.shape[1]})")


if __name__ == "__main__":
    main()
