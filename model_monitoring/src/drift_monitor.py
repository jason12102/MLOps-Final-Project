import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def run_drift_monitor():

    print("Running data drift monitoring...")

    # Load datasets
    reference_df = pd.read_csv("data/diabetes_reference.csv")
    current_df = pd.read_csv("data/current_data.csv")

    # Create Evidently report
    report = Report(metrics=[
        DataDriftPreset()
    ])

    # Run report
    report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    # Save report
    report.save_html("workspace/drift_report.html")

    print("Drift report saved to reports/drift_report.html")