"""
save_results.py
"""

import pandas as pd
from pathlib import Path


def save_training_results(results, filename):
    """
    Save training results to CSV.

    Args:
        results (list): List of dictionaries.
        filename (str): Output CSV filename.
    """

    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    save_path = results_dir / filename

    df.to_csv(save_path, index=False)

    print(f"✅ Results saved to {save_path}")

    return df