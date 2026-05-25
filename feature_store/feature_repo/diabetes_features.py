from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

patient = Entity(
    name="patient",
    join_keys=["patient_id"],
    value_type=ValueType.INT64,
    description="Synthetic patient id generated from row index.",
)

diabetes_source = FileSource(
    path="data/diabetes_features.parquet",
    event_timestamp_column="event_timestamp",
    timestamp_field="event_timestamp",
)

diabetes_profile_fv = FeatureView(
    name="diabetes_profile",
    entities=[patient],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="Pregnancies", dtype=Int64),
        Field(name="Glucose", dtype=Float32),
        Field(name="BloodPressure", dtype=Float32),
        Field(name="SkinThickness", dtype=Float32),
        Field(name="Insulin", dtype=Float32),
        Field(name="BMI", dtype=Float32),
        Field(name="DiabetesPedigreeFunction", dtype=Float32),
        Field(name="Age", dtype=Int64),
        Field(name="BMI_category", dtype=String),
        Field(name="Age_group", dtype=String),
        Field(name="Glucose_BMI", dtype=Float32),
    ],
    source=diabetes_source,
    online=True,
)

diabetes_prediction_features = FeatureService(
    name="diabetes_prediction_features",
    features=[diabetes_profile_fv],
)
