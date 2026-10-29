"""Accuracy alert check."""

import datetime
import json
from typing import Any

import pandas as pd

from hotel_demo.paths import data_dir

CHECK_NAME = "accuracy"
THRESHOLD = 0.60


def _matured_booking_ids(curr_date: datetime.date) -> pd.Series:
    """Return the IDs of the bookings whose arrival date is before curr_date."""
    b_dir = data_dir() / "bookings"
    booking_files = sorted(b_dir.glob("*.parquet"))
    if not booking_files:
        return pd.Series(dtype="int64")
    bookings = pd.concat(
        [pd.read_parquet(f, columns=["booking_id", "arrival_date"]) for f in booking_files],
        ignore_index=True,
    )
    arrival = pd.to_datetime(bookings["arrival_date"]).dt.date
    return bookings.loc[arrival < curr_date, "booking_id"]


def run(date: str) -> dict[str, Any]:
    """Run the accuracy check for the given date.

    Measures accuracy over matured labels released in the last 30 days, joined on
    booking_id to their predictions. Fires if value < 0.60.
    A label is matured when its booking's arrival date (from the bookings files) is
    before the given date: the outcome is then final. Cancellations are labelled as
    soon as they happen, so labels of bookings not yet arrived are mostly cancellations
    and would skew the reading; a label whose booking is not in the bookings files is
    not matured either.
    Appends the result record to data_dir() / "monitoring" / f"{date}.jsonl".
    """
    curr_date = datetime.date.fromisoformat(date)
    start_date = curr_date - datetime.timedelta(days=29)

    l_dir = data_dir() / "labels"
    label_files = sorted(l_dir.glob("*.parquet"))

    window_labels = pd.DataFrame()
    if label_files:
        all_labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
        all_labels["label_date_dt"] = pd.to_datetime(all_labels["label_date"]).dt.date
        window_labels = all_labels[
            (all_labels["label_date_dt"] >= start_date) & (all_labels["label_date_dt"] <= curr_date)
        ].copy()

    if len(window_labels) > 0:
        window_labels = window_labels[
            window_labels["booking_id"].isin(_matured_booking_ids(curr_date))
        ]

    p_dir = data_dir() / "predictions"
    pred_files = sorted(p_dir.glob("*.parquet"))
    all_preds = pd.DataFrame()
    if pred_files:
        all_preds = pd.concat([pd.read_parquet(pf) for pf in pred_files], ignore_index=True)

    merged = pd.DataFrame()
    if len(window_labels) > 0 and len(all_preds) > 0:
        merged = window_labels[["booking_id", "is_canceled"]].merge(
            all_preds[["booking_id", "p_cancel"]],
            on="booking_id",
            how="inner",
        )
        merged = merged.drop_duplicates(subset=["booking_id"], keep="last")

    if len(merged) == 0:
        fired = False
        value = None
        detail: Any = "no data"
    else:
        y_true = merged["is_canceled"].astype(int)
        y_pred = (merged["p_cancel"] >= 0.5).astype(int)
        total = len(merged)
        correct = int((y_true == y_pred).sum())
        value = float(correct / total)
        fired = bool(value < THRESHOLD)
        detail = {
            "total_evaluated": total,
            "correct": correct,
        }

    record = {
        "check": CHECK_NAME,
        "fired": fired,
        "value": value,
        "threshold": float(THRESHOLD),
        "detail": detail,
    }

    mon_dir = data_dir() / "monitoring"
    mon_dir.mkdir(parents=True, exist_ok=True)
    mon_file = mon_dir / f"{date}.jsonl"
    with mon_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record
