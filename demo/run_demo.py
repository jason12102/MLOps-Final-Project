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
    Create modified test data by:
    1. Swapping Glucose and BloodPressure values
    2. Adding noise to BMI values
    """
    modified = df.copy()
    
    # Modification 1: Swap Glucose and BloodPressure columns
    print("  [MOD 1] Swapping Glucose <-> BloodPressure values")
    modified["Glucose"], modified["BloodPressure"] = (
        df["BloodPressure"].values.copy(), 
        df["Glucose"].values.copy()
    )
    
    # Modification 2: Multiply Insulin by 2 (simulate measurement change)
    print("  [MOD 2] Doubling Insulin values (simulating sensor drift)")
    modified["Insulin"] = df["Insulin"] * 2
    
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
    for col in ["Glucose", "BloodPressure", "Insulin"]:
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
    original_metrics = None
    modified_metrics = None
    modified_df = None
    
    if args.step in ["original", "all"]:
        original_test_df, _, original_metrics = run_original_data()
    
    if args.step in ["modified", "all"]:
        modified_df, _, modified_metrics = run_modified_data(original_test_df)
    
    if args.step == "all" and original_metrics and modified_metrics:
        compare_results(original_metrics, modified_metrics)
        
        if original_test_df is not None and modified_df is not None:
            save_for_monitoring(original_test_df, modified_df)
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
