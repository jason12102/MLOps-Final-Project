import argparse
import json
from pathlib import Path

from feast import FeatureStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch online features from Feast.")
    parser.add_argument(
        "--patient-ids",
        nargs="+",
        type=int,
        required=True,
        help="Patient ids to query from the online store.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    repo_path = project_root / "feature_store" / "feature_repo"

    store = FeatureStore(repo_path=str(repo_path))

    feature_refs = [
        "diabetes_profile:Pregnancies",
        "diabetes_profile:Glucose",
        "diabetes_profile:BloodPressure",
        "diabetes_profile:SkinThickness",
        "diabetes_profile:Insulin",
        "diabetes_profile:BMI",
        "diabetes_profile:DiabetesPedigreeFunction",
        "diabetes_profile:Age",
        "diabetes_profile:BMI_category",
        "diabetes_profile:Age_group",
        "diabetes_profile:Glucose_BMI",
    ]

    entity_rows = [{"patient_id": pid} for pid in args.patient_ids]
    result = store.get_online_features(features=feature_refs, entity_rows=entity_rows)
    print(json.dumps(result.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()

