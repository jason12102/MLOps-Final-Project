#!/usr/bin/env python3
"""
Demo Script: Model Validation and Drift Detection

This script demonstrates:
1. Running original test data through the model and getting metrics
2. Creating modified test data (swapping/changing features)
3. Running modified data and observing drift detection

Usage:
    python run_demo.py --step original    # Run with original test data
    python run_demo.py --step modified    # Run with modified test data  
    python run_demo.py --step all         # Run both and compare
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report,
)
from evidently import Dataset, DataDefinition, BinaryClassification, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import Workspace

API_URL = "http://localhost:8000"
DATA_PATH = Path(__file__).parent.parent / "data_versioning" / "diabetes.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"

BMI_BINS = [-float("inf"), 18.5, 25, 30, float("inf")]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]
AGE_BINS = [-float("inf"), 29, 39, 49, 59, float("inf")]
AGE_LABELS = ["20s", "30s", "40s", "50s", "60+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to match model expectations."""
    df = df.copy()
    
    # Replace zeros with median for columns where 0 is invalid (missing value indicator)
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zeros:
        median_val = df.loc[df[col] > 0, col].median()
        df[col] = df[col].replace(0, median_val)
    
    df["BMI_category"] = pd.cut(df["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False).astype(str)
    df["Age_group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    return df


def check_api_health() -> bool:
    """Check if model serving API is available."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        data = resp.json()
        return data.get("model_loaded", False)
    except Exception as e:
        print(f"API not available: {e}")
        return False


def get_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Get predictions from the model serving API."""
    feature_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        "BMI_category", "Age_group", "Glucose_BMI"
    ]
    
    records = df[feature_cols].to_dict(orient="records")
    
    resp = requests.post(
        f"{API_URL}/predict/batch",
        json=records,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    
    result = resp.json()
    predictions = result["predictions"]
    
    df = df.copy()
    df["prediction"] = [p["prediction"] for p in predictions]
    df["prob_diabetes"] = [p["probability_diabetes"] for p in predictions]
    df["prob_no_diabetes"] = [p["probability_no_diabetes"] for p in predictions]
    
    return df, result.get("model_id")


def calculate_metrics(y_true, y_pred, y_prob) -> dict:
    """Calculate classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def print_metrics(metrics: dict, title: str):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for name, value in metrics.items():
        print(f"  {name.upper():12}: {value:.4f}")
    print()


def create_modified_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create modified test data with aggressive drift to trigger detection.
    Applies modifications to majority of columns to exceed 0.5 drift threshold.
    """
    modified = df.copy()
    
    # Modification 1: Swap Glucose and BloodPressure columns
    print("  [MOD 1] Swapping Glucose <-> BloodPressure values")
    modified["Glucose"], modified["BloodPressure"] = (
        df["BloodPressure"].values.copy(), 
        df["Glucose"].values.copy()
    )
    
    # Modification 2: Multiply Insulin by 3x (more aggressive)
    print("  [MOD 2] Tripling Insulin values (simulating sensor drift)")
    modified["Insulin"] = df["Insulin"] * 3
    
    # Modification 3: Shift BMI distribution significantly
    print("  [MOD 3] Shifting BMI by +10 (simulating population change)")
    modified["BMI"] = df["BMI"] + 10
    
    # Modification 4: Increase Age by 15 years
    print("  [MOD 4] Increasing Age by 15 years (simulating demographic shift)")
    modified["Age"] = df["Age"] + 15
    
    # Modification 5: Double SkinThickness
    print("  [MOD 5] Doubling SkinThickness values")
    modified["SkinThickness"] = df["SkinThickness"] * 2
    
    # Modification 6: Shift Pregnancies
    print("  [MOD 6] Adding 2 to Pregnancies count")
    modified["Pregnancies"] = df["Pregnancies"] + 2
    
    # Recalculate derived features with modified values
    modified["Glucose_BMI"] = modified["Glucose"] * modified["BMI"]
    modified["BMI_category"] = pd.cut(
        modified["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False
    ).astype(str)
    modified["Age_group"] = pd.cut(
        modified["Age"], bins=AGE_BINS, labels=AGE_LABELS
    ).astype(str)
    
    return modified


def run_original_data():
    """Run evaluation with original test data."""
    print("\n" + "="*60)
    print("  STEP 1: Original Test Data Evaluation")
    print("="*60)
    
    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)
    
    # Use 20% as test set
    test_df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    print(f"\nTest set size: {len(test_df)} samples")
    
    # Get predictions
    print("Getting predictions from model...")
    pred_df, model_id = get_predictions(test_df)
    print(f"Model ID: {model_id}")
    
    # Calculate metrics
    metrics = calculate_metrics(
        pred_df["Outcome"].values,
        pred_df["prediction"].values,
        pred_df["prob_diabetes"].values,
    )
    
    print_metrics(metrics, "Original Data Metrics")
    
    # Show confusion matrix
    cm = confusion_matrix(pred_df["Outcome"], pred_df["prediction"])
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUTPUT_DIR / "original_predictions.csv", index=False)
    
    with open(OUTPUT_DIR / "original_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    return test_df, pred_df, metrics


def run_modified_data(original_test_df: pd.DataFrame = None):
    """Run evaluation with modified test data."""
    print("\n" + "="*60)
    print("  STEP 2: Modified Test Data Evaluation")
    print("="*60)
    
    if original_test_df is None:
        df = pd.read_csv(DATA_PATH)
        df = add_features(df)
        original_test_df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    
    print(f"\nCreating modified data ({len(original_test_df)} samples)...")
    modified_df = create_modified_data(original_test_df)
    
    # Show what changed
    print("\nFeature distribution changes:")
    for col in ["Glucose", "BloodPressure", "Insulin", "BMI", "Age", "SkinThickness", "Pregnancies"]:
        orig_mean = original_test_df[col].mean()
        mod_mean = modified_df[col].mean()
        print(f"  {col:20}: {orig_mean:8.2f} -> {mod_mean:8.2f} (delta: {mod_mean - orig_mean:+.2f})")
    
    # Get predictions
    print("\nGetting predictions from model...")
    pred_df, model_id = get_predictions(modified_df)
    print(f"Model ID: {model_id}")
    
    # Calculate metrics (using ORIGINAL ground truth - this shows model degradation)
    metrics = calculate_metrics(
        original_test_df["Outcome"].values,  # Original ground truth
        pred_df["prediction"].values,
        pred_df["prob_diabetes"].values,
    )
    
    print_metrics(metrics, "Modified Data Metrics (vs Original Ground Truth)")
    
    # Show confusion matrix
    cm = confusion_matrix(original_test_df["Outcome"], pred_df["prediction"])
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modified_df.to_csv(OUTPUT_DIR / "modified_data.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / "modified_predictions.csv", index=False)
    
    with open(OUTPUT_DIR / "modified_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    return modified_df, pred_df, metrics


def compare_results(original_metrics: dict, modified_metrics: dict):
    """Compare original vs modified metrics."""
    print("\n" + "="*60)
    print("  COMPARISON: Original vs Modified")
    print("="*60)
    print(f"\n{'Metric':<12} {'Original':>10} {'Modified':>10} {'Delta':>10} {'Status':<10}")
    print("-" * 54)
    
    for metric in original_metrics:
        orig = original_metrics[metric]
        mod = modified_metrics[metric]
        delta = mod - orig
        status = "OK" if delta >= -0.05 else "DEGRADED"
        print(f"{metric.upper():<12} {orig:>10.4f} {mod:>10.4f} {delta:>+10.4f} {status:<10}")
    
    print("\n" + "="*60)
    print("  DRIFT DETECTION SUMMARY")
    print("="*60)
    print("""
  The modified data should trigger drift detection because:
  
  1. Glucose <-> BloodPressure SWAP:
     - Glucose values now have BloodPressure distribution (lower values)
     - BloodPressure values now have Glucose distribution (higher values)
     - This causes FEATURE DRIFT on both columns
  
  2. Insulin values DOUBLED:
     - Distribution shifted significantly
     - This causes FEATURE DRIFT on Insulin column
  
  3. Model performance likely DEGRADED:
     - Model was trained on original feature relationships
     - Swapped features break learned correlations
     - Accuracy/AUC should be noticeably worse
  
  To see drift in Evidently UI:
  1. Run: docker-compose up
  2. Open: http://localhost:8001
  3. View the drift report comparing reference vs modified data
""")


def save_for_monitoring(original_df: pd.DataFrame, modified_df: pd.DataFrame):
    """Save data files for the monitoring service to pick up."""
    monitoring_data_dir = Path(__file__).parent.parent / "model_monitoring" / "workspace"
    monitoring_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as current data for monitoring
    original_df.to_csv(monitoring_data_dir / "current_data_original.csv", index=False)
    modified_df.to_csv(monitoring_data_dir / "current_data_modified.csv", index=False)
    
    # Also save modified as the "current" data for automatic drift detection
    modified_df.to_csv(monitoring_data_dir / "current_data.csv", index=False)
    
    print(f"\nMonitoring data saved to {monitoring_data_dir}/")
    print("  - current_data_original.csv (baseline)")
    print("  - current_data_modified.csv (with drift)")
    print("  - current_data.csv (active - set to modified for demo)")


def create_evidently_reports(
    reference_df: pd.DataFrame, 
    current_df: pd.DataFrame,
    ref_predictions: pd.DataFrame,
    curr_predictions: pd.DataFrame,
):
    """Create Evidently reports and save to workspace for UI."""
    print("\n" + "="*60)
    print("  Creating Evidently Reports for UI")
    print("="*60)
    
    workspace_path = Path(__file__).parent.parent / "model_monitoring" / "workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    ws = Workspace.create(str(workspace_path))
    
    project_name = "Diabetes Model Monitoring"
    existing_projects = [p for p in ws.list_projects() if p.name == project_name]
    
    if existing_projects:
        project = existing_projects[0]
        print(f"  Using existing project: {project_name}")
    else:
        project = ws.create_project(project_name)
        project.description = "Monitoring diabetes prediction model for data drift and performance"
        project.save()
        print(f"  Created new project: {project_name}")
    
    # Prepare reference data with predictions
    ref_data = reference_df.copy()
    ref_data["prediction"] = ref_predictions["prediction"].values
    ref_data["prob_diabetes"] = ref_predictions["prob_diabetes"].values
    
    # Prepare current data with predictions  
    curr_data = current_df.copy()
    curr_data["prediction"] = curr_predictions["prediction"].values
    curr_data["prob_diabetes"] = curr_predictions["prob_diabetes"].values
    
    # Define data schema for Evidently 0.7+
    data_definition = DataDefinition(
        numerical_columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Glucose_BMI"
        ],
        categorical_columns=["BMI_category", "Age_group"],
        classification=[BinaryClassification(target="Outcome", prediction_labels="prediction")],
    )
    
    # Create datasets
    ref_dataset = Dataset.from_pandas(ref_data, data_definition=data_definition)
    curr_dataset = Dataset.from_pandas(curr_data, data_definition=data_definition)
    
    # Create CONTROL report (reference vs reference - no drift expected)
    print("  Generating Control Report (reference vs reference)...")
    control_report = Report([DataDriftPreset()])
    control_snapshot = control_report.run(reference_data=ref_dataset, current_data=ref_dataset)
    ws.add_run(project.id, control_snapshot, name="[CONTROL] Baseline - No Drift Expected")
    
    # Create Data Drift Report (reference vs modified - drift expected)
    print("  Generating Data Drift Report (reference vs modified)...")
    drift_report = Report([DataDriftPreset()])
    drift_snapshot = drift_report.run(reference_data=ref_dataset, current_data=curr_dataset)
    ws.add_run(project.id, drift_snapshot, name="[DRIFT] Modified Data - Drift Detected")
    
    # Create Data Summary Report for modified data
    print("  Generating Data Summary Report...")
    summary_report = Report([DataSummaryPreset()])
    summary_snapshot = summary_report.run(reference_data=ref_dataset, current_data=curr_dataset)
    ws.add_run(project.id, summary_snapshot, name="[SUMMARY] Data Statistics Comparison")
    
    print(f"\n  Reports saved to Evidently UI!")
    print(f"  Open http://localhost:8001 to view the reports")
    print(f"  Reports tagged:")
    print(f"    - [CONTROL] Baseline - No Drift Expected")
    print(f"    - [DRIFT] Modified Data - Drift Detected")
    print(f"    - [SUMMARY] Data Statistics Comparison")


def main():
    parser = argparse.ArgumentParser(description="Model Validation Demo")
    parser.add_argument(
        "--step",
        choices=["original", "modified", "all"],
        default="all",
        help="Which step to run",
    )
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  DIABETES MODEL VALIDATION DEMO")
    print("#"*60)
    
    # Check API
    if not check_api_health():
        print("\nERROR: Model serving API not available!")
        print("Start it with: docker-compose up")
        sys.exit(1)
    
    print("\nAPI Status: HEALTHY")
    
    original_test_df = None
    original_pred_df = None
    original_metrics = None
    modified_df = None
    modified_pred_df = None
    modified_metrics = None
    
    if args.step in ["original", "all"]:
        original_test_df, original_pred_df, original_metrics = run_original_data()
    
    if args.step in ["modified", "all"]:
        modified_df, modified_pred_df, modified_metrics = run_modified_data(original_test_df)
    
    if args.step == "all" and original_metrics and modified_metrics:
        compare_results(original_metrics, modified_metrics)
        
        if original_test_df is not None and modified_df is not None:
            save_for_monitoring(original_test_df, modified_df)
            
            # Create Evidently reports for UI
            create_evidently_reports(
                reference_df=original_test_df,
                current_df=modified_df,
                ref_predictions=original_pred_df,
                curr_predictions=modified_pred_df,
            )
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
