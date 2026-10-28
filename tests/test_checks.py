import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from hotel_demo.paths import data_dir
from hotel_demo.pipeline import deploy, train


@pytest.fixture
def checks_env(tmp_path, monkeypatch):
    data_target = tmp_path / "data"
    mlflow_target = tmp_path / "mlflow"
    fixture_data = Path(__file__).parent / "fixtures" / "data"

    shutil.copytree(fixture_data, data_target)
    mlflow_target.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOTEL_DEMO_DATA", str(data_target))
    monkeypatch.setenv("HOTEL_DEMO_MLFLOW", str(mlflow_target))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    v1 = train(cutoff="2026-01-10", min_child_samples=5)
    deploy(version=v1)

    return {"data": data_target, "mlflow": mlflow_target, "version": v1}


def _read_monitoring_lines(date: str) -> list[dict]:
    mon_file = data_dir() / "monitoring" / f"{date}.jsonl"
    assert mon_file.exists(), f"Monitoring file {mon_file} does not exist"
    lines = []
    with mon_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines


def test_cancellation_rate_check_fires_and_stays_quiet(checks_env):
    from hotel_demo.checks import cancellation_rate

    assert cancellation_rate.CHECK_NAME == "cancellation_rate"

    pred_dir = data_dir() / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 1. Quiet scenario: p_cancel < 0.5 for all (share = 0.0 <= 0.30)
    df_quiet = pd.DataFrame(
        {
            "booking_id": [1, 2, 3, 4],
            "business_date": "2026-01-11",
            "model_version": "1",
            "p_cancel": [0.1, 0.2, 0.3, 0.4],
        }
    )
    df_quiet.to_parquet(pred_dir / "2026-01-11.parquet", index=False)

    res_quiet = cancellation_rate.run("2026-01-11")
    assert set(res_quiet.keys()) == {"check", "fired", "value", "threshold", "detail"}
    assert res_quiet["check"] == "cancellation_rate"
    assert res_quiet["fired"] is False
    assert res_quiet["value"] == 0.0
    assert res_quiet["threshold"] == 0.30

    lines = _read_monitoring_lines("2026-01-11")
    assert len(lines) == 1
    assert set(lines[0].keys()) == {"check", "fired", "value", "threshold", "detail"}
    assert lines[0] == res_quiet

    # 2. Firing scenario: p_cancel >= 0.5 for 3 out of 4 (share = 0.75 > 0.30)
    # Remove previous quiet file so only the new day's predictions are evaluated
    (pred_dir / "2026-01-11.parquet").unlink()
    df_fire = pd.DataFrame(
        {
            "booking_id": [5, 6, 7, 8],
            "business_date": "2026-01-12",
            "model_version": "1",
            "p_cancel": [0.8, 0.9, 0.7, 0.2],
        }
    )
    df_fire.to_parquet(pred_dir / "2026-01-12.parquet", index=False)

    res_fire = cancellation_rate.run("2026-01-12")
    assert res_fire["fired"] is True
    assert res_fire["value"] == 0.75
    assert res_fire["threshold"] == 0.30


def test_drift_check_fires_and_stays_quiet(checks_env):
    from hotel_demo.checks import drift

    assert drift.CHECK_NAME == "drift"

    # 1. Quiet scenario: bookings distribution across 2026-01-11 to 2026-01-15 matches training
    res_quiet = drift.run("2026-01-15")
    assert set(res_quiet.keys()) == {"check", "fired", "value", "threshold", "detail"}
    assert res_quiet["check"] == "drift"
    assert res_quiet["threshold"] == 0.15
    # The normal fixture does not drift significantly
    assert res_quiet["fired"] is False
    assert res_quiet["value"] <= 0.15

    lines = _read_monitoring_lines("2026-01-15")
    assert len(lines) == 1
    assert set(lines[0].keys()) == {"check", "fired", "value", "threshold", "detail"}
    assert lines[0] == res_quiet

    # 2. Firing scenario: heavily drift adults and children on 2026-01-16
    b_file = data_dir() / "bookings" / "2026-01-11.parquet"
    df = pd.read_parquet(b_file)
    df["adults"] = 50  # Huge shift in adults
    df["children"] = 20  # Huge shift in children
    drift_file = data_dir() / "bookings" / "2026-01-16.parquet"
    df.to_parquet(drift_file, index=False)

    res_fire = drift.run("2026-01-16")
    assert res_fire["fired"] is True
    assert res_fire["value"] > 0.15
    assert res_fire["threshold"] == 0.15


def test_accuracy_check_fires_and_stays_quiet(checks_env):
    from hotel_demo.checks import accuracy

    assert accuracy.CHECK_NAME == "accuracy"

    pred_dir = data_dir() / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 1. High accuracy scenario (stays quiet, value >= 0.60)
    # Booking IDs in labels: let's match predictions exactly to labels
    l_df = pd.read_parquet(data_dir() / "labels" / "2026-01-12.parquet")
    # Make predictions perfectly match labels
    pred_rows = []
    for _, row in l_df.iterrows():
        b_id = int(row["booking_id"])
        is_canc = int(row["is_canceled"])
        p_canc = 0.9 if is_canc == 1 else 0.1
        pred_rows.append(
            {
                "booking_id": b_id,
                "business_date": "2026-01-11",
                "model_version": "1",
                "p_cancel": p_canc,
            }
        )
    pd.DataFrame(pred_rows).to_parquet(pred_dir / "2026-01-11.parquet", index=False)

    res_quiet = accuracy.run("2026-01-12")
    assert set(res_quiet.keys()) == {"check", "fired", "value", "threshold", "detail"}
    assert res_quiet["check"] == "accuracy"
    assert res_quiet["threshold"] == 0.60
    assert res_quiet["fired"] is False
    assert res_quiet["value"] == 1.0

    lines = _read_monitoring_lines("2026-01-12")
    assert len(lines) == 1
    assert set(lines[0].keys()) == {"check", "fired", "value", "threshold", "detail"}
    assert lines[0] == res_quiet

    # 2. Low accuracy scenario (fires, value < 0.60)
    # Invert predictions so accuracy is 0.0
    bad_pred_rows = []
    for _, row in l_df.iterrows():
        b_id = int(row["booking_id"])
        is_canc = int(row["is_canceled"])
        p_canc = 0.1 if is_canc == 1 else 0.9  # inverted!
        bad_pred_rows.append(
            {
                "booking_id": b_id,
                "business_date": "2026-01-13",
                "model_version": "1",
                "p_cancel": p_canc,
            }
        )
    pd.DataFrame(bad_pred_rows).to_parquet(pred_dir / "2026-01-13.parquet", index=False)

    res_fire = accuracy.run("2026-01-13")
    assert res_fire["fired"] is True
    assert res_fire["value"] < 0.60
    assert res_fire["threshold"] == 0.60


def test_accuracy_check_no_data_reports_no_data_and_does_not_fire(checks_env):
    from hotel_demo.checks import accuracy

    # Date with no predictions or labels in window
    res = accuracy.run("2025-01-01")
    assert res["check"] == "accuracy"
    assert res["fired"] is False
    assert res["value"] is None
    assert res["threshold"] == 0.60
    assert res["detail"] == "no data"
