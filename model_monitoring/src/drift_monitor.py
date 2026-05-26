from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from utils import load_config, get_predictions_from_api


def run_drift_monitor(current_df: pd.DataFrame | None = None, current_data_path: str | None = None) -> dict:
    """
    Run data drift monitoring comparing reference data to current data.
    
    Args:
        current_df: Current data to compare. If None, loads from file.
        current_data_path: Path to current data CSV. If None, uses default.
    
    Returns:
        Dictionary with drift detection results.
    """
    print("Running data drift monitoring...")
    config = load_config()
    
    reference_path = config["reference_data"]["path"]
    reference_df = pd.read_csv(reference_path)
    
    if current_df is None:
        if current_data_path:
            current_path = Path(current_data_path)
        else:
            current_path = Path(config["monitoring"]["report_output_dir"]) / "current_data.csv"
        
        if current_path.exists():
            current_df = pd.read_csv(current_path)
            print(f"Loaded current data from {current_path}")
        else:
            print("No current data found. Using sample from reference for demo.")
            current_df = reference_df.sample(frac=0.3, random_state=42)
    
    feature_cols = (
        config["feature_columns"]["numerical"] + 
        config["feature_columns"]["categorical"]
    )
    target_col = config["target_column"]
    cols_to_use = [c for c in feature_cols + [target_col] if c in reference_df.columns]
    
    ref_subset = reference_df[cols_to_use].copy()
    curr_subset = current_df[[c for c in cols_to_use if c in current_df.columns]].copy()
    
    if target_col in reference_df.columns and target_col in current_df.columns:
        ref_with_pred, _ = get_predictions_from_api(ref_subset.drop(columns=[target_col]), config)
        curr_with_pred, _ = get_predictions_from_api(curr_subset.drop(columns=[target_col]), config)
        
        if ref_with_pred is not None:
            ref_subset["prediction"] = ref_with_pred["prediction"]
        if curr_with_pred is not None:
            curr_subset["prediction"] = curr_with_pred["prediction"]
    
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])
    
    report.run(
        reference_data=ref_subset,
        current_data=curr_subset,
    )
    
    output_dir = Path(config["monitoring"]["report_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))
    
    latest_path = output_dir / "drift_report_latest.html"
    report.save_html(str(latest_path))
    
    print(f"Drift report saved to {report_path}")
    
    report_dict = report.as_dict()
    drift_detected = False
    drift_share = 0.0
    
    for metric in report_dict.get("metrics", []):
        if "data_drift" in str(metric.get("metric", "")).lower():
            result = metric.get("result", {})
            drift_share = result.get("share_of_drifted_columns", 0)
            drift_detected = drift_share > config["monitoring"]["drift_threshold"]
            break
    
    return {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "threshold": config["monitoring"]["drift_threshold"],
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    result = run_drift_monitor()
    print(f"Drift detected: {result['drift_detected']}")
    print(f"Drift share: {result['drift_share']:.2%}")
