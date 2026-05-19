import pandas as pd
from sklearn.metrics import accuracy_score

def run_performance_monitor():

    df = pd.read_csv("data/current_data.csv")

    accuracy = accuracy_score(
        df["target"],
        df["prediction"]
    )

    print(f"Model Accuracy: {accuracy:.4f}")