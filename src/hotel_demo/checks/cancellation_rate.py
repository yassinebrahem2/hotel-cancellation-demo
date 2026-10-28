"""Cancellation-rate alert check."""

import datetime
import json
from typing import Any

import pandas as pd

from hotel_demo.paths import data_dir

CHECK_NAME = "cancellation_rate"
THRESHOLD = 0.30


def run(date: str) -> dict[str, Any]:
    """Run the cancellation rate check for the given date.

    Measures the share of p_cancel >= 0.5 over the last 7 days of predictions.
    Appends the result record to data_dir() / "monitoring" / f"{date}.jsonl".
    """
    curr_date = datetime.date.fromisoformat(date)
    pred_dir = data_dir() / "predictions"

    # Prediction file for the current date is required
    today_file = pred_dir / f"{date}.parquet"
    if not today_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {today_file}")

    target_dates = [curr_date - datetime.timedelta(days=i) for i in range(7)]
    dfs: list[pd.DataFrame] = []
    days_evaluated: list[str] = []

    for d in reversed(target_dates):
        f = pred_dir / f"{d.isoformat()}.parquet"
        if f.exists():
            dfs.append(pd.read_parquet(f))
            days_evaluated.append(d.isoformat())

    all_preds = pd.concat(dfs, ignore_index=True)
    total = len(all_preds)
    high_risk = int((all_preds["p_cancel"] >= 0.5).sum())
    value = float((all_preds["p_cancel"] >= 0.5).mean()) if total > 0 else 0.0
    fired = bool(value > THRESHOLD)

    record = {
        "check": CHECK_NAME,
        "fired": fired,
        "value": float(value),
        "threshold": float(THRESHOLD),
        "detail": {
            "total_predictions": total,
            "high_risk_predictions": high_risk,
            "days_evaluated": days_evaluated,
        },
    }

    mon_dir = data_dir() / "monitoring"
    mon_dir.mkdir(parents=True, exist_ok=True)
    mon_file = mon_dir / f"{date}.jsonl"
    with mon_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record
