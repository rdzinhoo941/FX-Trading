import pandas as pd
import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

def load_nav_series(model_name: str):
    """
    Load NAV time series for a given model from CSV.
    Expected path:
    data/results/<model_name>/nav.csv
    """

    file_path = os.path.join(
        BASE_PATH,
        "data",
        "results",
        model_name,
        "nav.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NAV file not found: {file_path}")

    df = pd.read_csv(file_path)


    return [
        {
            "date": str(row["date"]),
            "value": float(row["nav"])
        }
        for _, row in df.iterrows()
    ]
