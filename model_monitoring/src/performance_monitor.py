from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import (
    ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix,
)
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from utils import load_config, get_predictions_from_api, check_api_health


def run_performance_monitor(current_df: pd.DataFrame | None = None) -> dict:
    """
    Run model performance monitoring.
    
    Args:
        current_df: Current data with ground truth. If None, uses reference data sample.
    
    Returns:
        Dictionary with performance metrics and alerts.
    """
    print("Running performance monitoring...")
    config = load_config()
    
    if not check_api_health(config):
        print("Warning: Model serving API is not healthy")
        return {"error": "API not available"}
    
    target_col = config["target_column"]
    
    if current_df is None:
        reference_path = config["reference_data"]["path"]
        reference_df = pd.read_csv(reference_path)
        current_df = reference_df.sample(frac=0.3, random_state=42)
    
    if target_col not in current_df.columns:
        print(f"Warning: Target column '{target_col}' not in current data")
        return {"error": "No ground truth available"}
    
    feature_df = current_df.drop(columns=[target_col])
    pred_df, api_response = get_predictions_from_api(feature_df, config)
    
    if pred_df is None:
        return {"error": "Failed to get predictions from API"}
    
    y_true = current_df[target_col].values
    y_pred = pred_df["prediction"].values
    y_prob = pred_df["probability_diabetes"].values
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
    
    thresholds = config["monitoring"]["performance_threshold"]
    alerts = []
    
    if metrics["accuracy"] < thresholds.get("accuracy", 0):
        alerts.append(f"Accuracy {metrics['accuracy']:.4f} below threshold {thresholds['accuracy']}")
    if metrics["auc"] < thresholds.get("auc", 0):
        alerts.append(f"AUC {metrics['auc']:.4f} below threshold {thresholds['auc']}")
    
    eval_df = current_df.copy()
    eval_df["prediction"] = y_pred
    eval_df["target"] = y_true
    
    reference_path = config["reference_data"]["path"]
    reference_df = pd.read_csv(reference_path)
    ref_features = reference_df.drop(columns=[target_col])
    ref_pred_df, _ = get_predictions_from_api(ref_features, config)
    
    if ref_pred_df is not None:
        ref_eval_df = reference_df.copy()
        ref_eval_df["prediction"] = ref_pred_df["prediction"].values
        ref_eval_df["target"] = reference_df[target_col].values
        
        report = Report(metrics=[
            ClassificationPreset(),
        ])
        
        report.run(
            reference_data=ref_eval_df,
            current_data=eval_df,
        )
    else:
        report = Report(metrics=[
            ClassificationQualityMetric(),
            ClassificationClassBalance(),
            ClassificationConfusionMatrix(),
        ])
        
        report.run(current_data=eval_df)
    
    output_dir = Path(config["monitoring"]["report_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"performance_report_{timestamp}.html"
    report.save_html(str(report_path))
    
    latest_path = output_dir / "performance_report_latest.html"
    report.save_html(str(latest_path))
    
    print(f"Performance report saved to {report_path}")
    
    print("\n=== Performance Metrics ===")
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value:.4f}")
    
    if alerts:
        print("\n=== ALERTS ===")
        for alert in alerts:
            print(f"  [!] {alert}")
    
    return {
        "metrics": metrics,
        "alerts": alerts,
        "report_path": str(report_path),
        "model_id": api_response.get("model_id") if api_response else None,
    }


if __name__ == "__main__":
    result = run_performance_monitor()
    if "error" not in result:
        print(f"\nModel: {result.get('model_id')}")
        print(f"Alerts: {len(result.get('alerts', []))}")
