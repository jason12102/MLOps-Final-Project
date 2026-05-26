import argparse
import time
from datetime import datetime

from utils import load_config, check_api_health
from drift_monitor import run_drift_monitor
from performance_monitor import run_performance_monitor


def wait_for_api(config: dict, max_retries: int = 30, delay: int = 10) -> bool:
    """Wait for the model serving API to become available."""
    print(f"Waiting for model serving API at {config['model_serving']['url']}...")
    
    for i in range(max_retries):
        if check_api_health(config):
            print("API is ready!")
            return True
        print(f"  Attempt {i+1}/{max_retries} - API not ready, waiting {delay}s...")
        time.sleep(delay)
    
    print("API did not become available")
    return False


def run_all_monitors() -> dict:
    """Run all monitoring checks."""
    print(f"\n{'='*60}")
    print(f"  Model Monitoring Run - {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    results = {}
    
    print("\n--- Drift Monitoring ---")
    try:
        results["drift"] = run_drift_monitor()
    except Exception as e:
        print(f"Drift monitoring failed: {e}")
        results["drift"] = {"error": str(e)}
    
    print("\n--- Performance Monitoring ---")
    try:
        results["performance"] = run_performance_monitor()
    except Exception as e:
        print(f"Performance monitoring failed: {e}")
        results["performance"] = {"error": str(e)}
    
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    
    if "error" not in results.get("drift", {}):
        drift = results["drift"]
        status = "DETECTED" if drift.get("drift_detected") else "OK"
        print(f"  Drift: {status} ({drift.get('drift_share', 0):.1%} columns drifted)")
    
    if "error" not in results.get("performance", {}):
        perf = results["performance"]
        metrics = perf.get("metrics", {})
        alerts = perf.get("alerts", [])
        print(f"  Performance: AUC={metrics.get('auc', 0):.4f}, Accuracy={metrics.get('accuracy', 0):.4f}")
        if alerts:
            print(f"  Alerts: {len(alerts)} active")
    
    return results


def run_scheduled(interval_minutes: int):
    """Run monitoring on a schedule."""
    print(f"Starting scheduled monitoring (every {interval_minutes} minutes)")
    
    while True:
        try:
            run_all_monitors()
        except Exception as e:
            print(f"Monitoring run failed: {e}")
        
        print(f"\nNext run in {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(description="Model Monitoring Service")
    parser.add_argument(
        "--mode",
        choices=["once", "scheduled", "wait-and-run"],
        default="once",
        help="Run mode: once, scheduled, or wait-and-run",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Interval in minutes for scheduled mode",
    )
    args = parser.parse_args()
    
    config = load_config()
    
    if args.mode in ["scheduled", "wait-and-run"]:
        if not wait_for_api(config):
            print("Exiting: API not available")
            return
    
    if args.mode == "once":
        if check_api_health(config):
            run_all_monitors()
        else:
            print("API not available. Run with --mode wait-and-run to wait for API.")
    
    elif args.mode == "wait-and-run":
        run_all_monitors()
    
    elif args.mode == "scheduled":
        interval = args.interval or config["schedule"]["interval_minutes"]
        run_scheduled(interval)


if __name__ == "__main__":
    main()
