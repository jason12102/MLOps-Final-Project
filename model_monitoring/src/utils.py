from pathlib import Path

import pandas as pd
import requests
import yaml


def load_config(config_path: str = "/app/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_api_health(config: dict) -> bool:
    """Check if the model serving API is healthy."""
    url = f"{config['model_serving']['url']}{config['model_serving']['health_endpoint']}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False)
    except requests.RequestException as e:
        print(f"API health check failed: {e}")
    return False


def get_predictions_from_api(
    df: pd.DataFrame, 
    config: dict
) -> tuple[pd.DataFrame | None, dict | None]:
    """
    Get predictions from the model serving API.
    
    Args:
        df: DataFrame with feature columns (no target).
        config: Configuration dictionary.
    
    Returns:
        Tuple of (DataFrame with predictions, raw API response).
    """
    url = f"{config['model_serving']['url']}{config['model_serving']['batch_endpoint']}"
    
    feature_cols = (
        config["feature_columns"]["numerical"] + 
        config["feature_columns"]["categorical"]
    )
    
    available_cols = [c for c in feature_cols if c in df.columns]
    
    records = df[available_cols].to_dict(orient="records")
    
    try:
        response = requests.post(
            url,
            json=records,
            timeout=120,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        
        result = response.json()
        predictions = result.get("predictions", [])
        
        pred_df = df.copy()
        pred_df["prediction"] = [p["prediction"] for p in predictions]
        pred_df["probability_diabetes"] = [p["probability_diabetes"] for p in predictions]
        pred_df["probability_no_diabetes"] = [p["probability_no_diabetes"] for p in predictions]
        
        return pred_df, result
        
    except requests.RequestException as e:
        print(f"API prediction request failed: {e}")
        return None, None


def save_current_data(df: pd.DataFrame, config: dict) -> Path:
    """Save current data for monitoring."""
    output_dir = Path(config["monitoring"]["report_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path = output_dir / "current_data.csv"
    df.to_csv(path, index=False)
    return path
