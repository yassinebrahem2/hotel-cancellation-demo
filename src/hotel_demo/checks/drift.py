"""Drift report check using two-sample Kolmogorov-Smirnov test."""

import datetime
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy.stats import ks_2samp

from hotel_demo.features import MODEL_ALIAS, MODEL_NAME
from hotel_demo.monitoring import write_monitoring_line
from hotel_demo.paths import data_dir, tracking_uri

CHECK_NAME = "drift"
THRESHOLD = 0.15
DRIFT_FEATURES = [
    "adults",
    "stays_in_week_nights",
    "stays_in_weekend_nights",
    "children",
]


def run(date: str) -> dict[str, Any]:
    """Run Kolmogorov-Smirnov drift check for the given date.

    Tests adults, stays_in_week_nights, stays_in_weekend_nights, children of the
    last 7 days of bookings against the production model's training window.
    Appends the result record to data_dir() / "monitoring" / f"{date}.jsonl".
    """
    curr_date = datetime.date.fromisoformat(date)
    b_dir = data_dir() / "bookings"

    # 1. Retrieve production model's training window from MLflow
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    try:
        prod_mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except Exception as exc:
        raise RuntimeError(f"No production model found for {MODEL_NAME}@{MODEL_ALIAS}") from exc

    run_info = client.get_run(prod_mv.run_id)
    params = run_info.data.params
    min_arr_str = params.get("min_arrival_date")
    max_arr_str = params.get("max_arrival_date")
    cutoff_str = params.get("cutoff")

    # 2. Read all bookings to extract the training window data
    booking_files = sorted(b_dir.glob("*.parquet"))
    if not booking_files:
        raise FileNotFoundError(f"No bookings found in {b_dir}")

    all_bookings = pd.concat([pd.read_parquet(f) for f in booking_files], ignore_index=True)
    all_bookings["arrival_date_dt"] = pd.to_datetime(all_bookings["arrival_date"]).dt.date

    if min_arr_str and max_arr_str:
        min_arr = datetime.date.fromisoformat(min_arr_str)
        max_arr = datetime.date.fromisoformat(max_arr_str)
        train_bookings = all_bookings[
            (all_bookings["arrival_date_dt"] >= min_arr)
            & (all_bookings["arrival_date_dt"] <= max_arr)
        ]
    elif cutoff_str:
        cutoff_date = datetime.date.fromisoformat(cutoff_str)
        train_bookings = all_bookings[all_bookings["arrival_date_dt"] < cutoff_date]
    else:
        train_bookings = None

    # 3. Read last 7 days of bookings up to date
    target_dates = [curr_date - datetime.timedelta(days=i) for i in range(7)]
    recent_dfs: list[pd.DataFrame] = []
    days_evaluated: list[str] = []

    for d in reversed(target_dates):
        f = b_dir / f"{d.isoformat()}.parquet"
        if f.exists():
            recent_dfs.append(pd.read_parquet(f))
            days_evaluated.append(d.isoformat())

    if not recent_dfs:
        raise FileNotFoundError(f"No bookings found in the last 7 days up to {date}")

    curr_bookings = pd.concat(recent_dfs, ignore_index=True)

    # 4. Compute two-sample KS statistic for each feature; a feature with an empty
    # sample, or no training window at all, gets no statistic rather than an invented one.
    stats: dict[str, float | None] = {}
    for feat in DRIFT_FEATURES:
        s_curr = curr_bookings[feat].dropna().to_numpy()
        s_train = train_bookings[feat].dropna().to_numpy() if train_bookings is not None else []
        if len(s_curr) == 0 or len(s_train) == 0:
            stats[feat] = None
        else:
            res = ks_2samp(s_curr, s_train, nan_policy="omit")
            stats[feat] = float(res.statistic)

    computed = [s for s in stats.values() if s is not None]
    detail: dict[str, Any] = {"features": stats, "days_evaluated": days_evaluated}
    record: dict[str, Any]
    if computed:
        record = {
            "check": CHECK_NAME,
            "fired": bool(any(s > THRESHOLD for s in computed)),
            "value": float(max(computed)),
            "threshold": float(THRESHOLD),
            "detail": detail,
        }
    else:
        detail["reason"] = (
            "the production model's run has no training window"
            if train_bookings is None
            else "no feature has samples in both the current and the training window"
        )
        record = {
            "check": CHECK_NAME,
            "fired": False,
            "value": None,
            "not_enough_data": True,
            "threshold": float(THRESHOLD),
            "detail": detail,
        }

    return write_monitoring_line(date, record)
