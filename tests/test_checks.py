import datetime
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


def _assert_utc_iso_timestamp(
    ts: str,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
) -> None:
    dt = datetime.datetime.fromisoformat(ts)
    assert dt.tzinfo is not None, f"Expected timezone-aware timestamp, got {ts}"
    assert dt.utcoffset() == datetime.timedelta(0), f"Expected UTC offset 0, got {dt.utcoffset()}"
    if start is not None:
        assert start <= dt, f"Timestamp {dt} earlier than start {start}"
    if end is not None:
        assert dt <= end, f"Timestamp {dt} later than end {end}"


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

    start = datetime.datetime.now(datetime.UTC)
    res_quiet = cancellation_rate.run("2026-01-11")
    end = datetime.datetime.now(datetime.UTC)
    assert set(res_quiet.keys()) == {"check", "ran_at", "fired", "value", "threshold", "detail"}
    assert res_quiet["check"] == "cancellation_rate"
    assert res_quiet["fired"] is False
    assert res_quiet["value"] == 0.0
    assert res_quiet["threshold"] == 0.30
    _assert_utc_iso_timestamp(res_quiet["ran_at"], start, end)

    lines = _read_monitoring_lines("2026-01-11")
    assert len(lines) == 1
    assert set(lines[0].keys()) == {"check", "ran_at", "fired", "value", "threshold", "detail"}
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

    start_fire = datetime.datetime.now(datetime.UTC)
    res_fire = cancellation_rate.run("2026-01-12")
    end_fire = datetime.datetime.now(datetime.UTC)
    assert res_fire["fired"] is True
    assert res_fire["value"] == 0.75
    assert res_fire["threshold"] == 0.30
    _assert_utc_iso_timestamp(res_fire["ran_at"], start_fire, end_fire)


def test_drift_check_fires_and_stays_quiet(checks_env):
    from hotel_demo.checks import drift

    assert drift.CHECK_NAME == "drift"

    # 1. Quiet scenario: bookings distribution across 2026-01-11 to 2026-01-15 matches training
    start = datetime.datetime.now(datetime.UTC)
    res_quiet = drift.run("2026-01-15")
    end = datetime.datetime.now(datetime.UTC)
    assert set(res_quiet.keys()) == {"check", "ran_at", "fired", "value", "threshold", "detail"}
    assert res_quiet["check"] == "drift"
    assert res_quiet["threshold"] == 0.15
    # The normal fixture does not drift significantly
    assert res_quiet["fired"] is False
    assert res_quiet["value"] <= 0.15
    _assert_utc_iso_timestamp(res_quiet["ran_at"], start, end)

    lines = _read_monitoring_lines("2026-01-15")
    assert len(lines) == 1
    assert set(lines[0].keys()) == {"check", "ran_at", "fired", "value", "threshold", "detail"}
    assert lines[0] == res_quiet

    # 2. Firing scenario: heavily drift adults and children on 2026-01-16
    b_file = data_dir() / "bookings" / "2026-01-11.parquet"
    df = pd.read_parquet(b_file)
    df["adults"] = 50  # Huge shift in adults
    df["children"] = 20  # Huge shift in children
    drift_file = data_dir() / "bookings" / "2026-01-16.parquet"
    df.to_parquet(drift_file, index=False)

    start_fire = datetime.datetime.now(datetime.UTC)
    res_fire = drift.run("2026-01-16")
    end_fire = datetime.datetime.now(datetime.UTC)
    assert res_fire["fired"] is True
    assert res_fire["value"] > 0.15
    assert res_fire["threshold"] == 0.15
    _assert_utc_iso_timestamp(res_fire["ran_at"], start_fire, end_fire)


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

    start = datetime.datetime.now(datetime.UTC)
    res_quiet = accuracy.run("2026-01-12")
    end = datetime.datetime.now(datetime.UTC)
    assert set(res_quiet.keys()) == {"check", "ran_at", "fired", "value", "threshold", "detail"}
    assert res_quiet["check"] == "accuracy"
    assert res_quiet["threshold"] == 0.60
    assert res_quiet["fired"] is False
    assert res_quiet["value"] == 1.0
    _assert_utc_iso_timestamp(res_quiet["ran_at"], start, end)

    lines = _read_monitoring_lines("2026-01-12")
    assert len(lines) == 1
    assert set(lines[0].keys()) == {"check", "ran_at", "fired", "value", "threshold", "detail"}
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

    start_fire = datetime.datetime.now(datetime.UTC)
    res_fire = accuracy.run("2026-01-13")
    end_fire = datetime.datetime.now(datetime.UTC)
    assert res_fire["fired"] is True
    assert res_fire["value"] < 0.60
    assert res_fire["threshold"] == 0.60
    _assert_utc_iso_timestamp(res_fire["ran_at"], start_fire, end_fire)


def test_accuracy_check_no_data_reports_no_data_and_does_not_fire(checks_env):
    from hotel_demo.checks import accuracy

    # Date with no predictions or labels in window
    start = datetime.datetime.now(datetime.UTC)
    res = accuracy.run("2025-01-01")
    end = datetime.datetime.now(datetime.UTC)
    assert res["check"] == "accuracy"
    assert res["fired"] is False
    assert res["value"] is None
    assert res["threshold"] == 0.60
    assert res["detail"] == "no data"
    _assert_utc_iso_timestamp(res["ran_at"], start, end)


# Matured-label cases: synthetic booking IDs (>= 5000) and a check date whose 30-day
# label window holds no fixture label, so only the rows written here are evaluated.
MATURED_CHECK_DATE = "2026-03-01"
MATURED = "2026-02-10"  # arrival before the check date: label is final
NOT_MATURED = "2026-03-05"  # arrival after the check date: label may still change


def _write_accuracy_case(rows: list[tuple[int, str | None, bool]]) -> None:
    """Write labels released in the window, bookings and predictions for the rows.

    Each row is (booking_id, arrival_date or None for a booking absent from the
    bookings files, whether the prediction is correct). Every booking is canceled.
    """
    d = data_dir()
    pd.DataFrame(
        {
            "booking_id": [b for b, _, _ in rows],
            "is_canceled": 1,
            "label_date": "2026-02-20",
            "booking_changes": 0,
        }
    ).to_parquet(d / "labels" / "2026-02-20.parquet", index=False)
    pd.DataFrame(
        {
            "booking_id": [b for b, arrival, _ in rows if arrival is not None],
            "arrival_date": [arrival for _, arrival, _ in rows if arrival is not None],
        }
    ).to_parquet(d / "bookings" / "2026-02-01.parquet", index=False)
    pred_dir = d / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "booking_id": [b for b, _, _ in rows],
            "business_date": "2026-02-01",
            "model_version": "1",
            "p_cancel": [0.9 if correct else 0.1 for _, _, correct in rows],
        }
    ).to_parquet(pred_dir / "2026-02-01.parquet", index=False)


def test_accuracy_check_excludes_labels_not_yet_matured(checks_env):
    from hotel_demo.checks import accuracy

    _write_accuracy_case(
        [
            (5000, MATURED, True),
            (5001, NOT_MATURED, False),
            (5002, NOT_MATURED, False),
            (5003, NOT_MATURED, False),
        ]
    )

    res = accuracy.run(MATURED_CHECK_DATE)
    assert res["fired"] is False
    assert res["value"] == 1.0
    assert res["detail"] == {"total_evaluated": 1, "correct": 1}


def test_accuracy_check_fires_on_wrong_matured_labels(checks_env):
    from hotel_demo.checks import accuracy

    _write_accuracy_case(
        [
            (5000, MATURED, False),
            (5001, NOT_MATURED, True),
            (5002, NOT_MATURED, True),
            (5003, NOT_MATURED, True),
        ]
    )

    res = accuracy.run(MATURED_CHECK_DATE)
    assert res["fired"] is True
    assert res["value"] == 0.0
    assert res["detail"] == {"total_evaluated": 1, "correct": 0}


def test_accuracy_check_treats_arrival_day_as_not_matured(checks_env):
    from hotel_demo.checks import accuracy

    _write_accuracy_case([(5000, MATURED, True), (5001, MATURED_CHECK_DATE, False)])

    res = accuracy.run(MATURED_CHECK_DATE)
    assert res["fired"] is False
    assert res["value"] == 1.0
    assert res["detail"] == {"total_evaluated": 1, "correct": 1}


def test_accuracy_check_excludes_labels_of_unknown_bookings(checks_env):
    from hotel_demo.checks import accuracy

    _write_accuracy_case([(5000, MATURED, True), (5001, None, False)])

    res = accuracy.run(MATURED_CHECK_DATE)
    assert res["fired"] is False
    assert res["value"] == 1.0
    assert res["detail"] == {"total_evaluated": 1, "correct": 1}


def test_accuracy_check_without_matured_labels_reports_no_data(checks_env):
    from hotel_demo.checks import accuracy

    _write_accuracy_case([(5000, NOT_MATURED, False), (5001, MATURED_CHECK_DATE, True)])

    res = accuracy.run(MATURED_CHECK_DATE)
    assert res["fired"] is False
    assert res["value"] is None
    assert res["detail"] == "no data"


def _assert_not_enough_data(res: dict, check: str, date: str) -> None:
    assert res["check"] == check
    assert res["not_enough_data"] is True
    assert res["value"] is None
    assert res["fired"] is False
    assert isinstance(res["detail"]["reason"], str)
    assert "ran_at" in res
    _assert_utc_iso_timestamp(res["ran_at"])
    assert _read_monitoring_lines(date) == [res]


def test_cancellation_rate_check_with_zero_predictions_reports_not_enough_data(checks_env):
    from hotel_demo.checks import cancellation_rate

    pred_dir = data_dir() / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"booking_id": [], "p_cancel": []}).astype(
        {"booking_id": "int64", "p_cancel": "float64"}
    ).to_parquet(pred_dir / "2026-01-11.parquet", index=False)

    res = cancellation_rate.run("2026-01-11")

    _assert_not_enough_data(res, "cancellation_rate", "2026-01-11")
    assert res["threshold"] == 0.30
    assert res["detail"]["total_predictions"] == 0


def test_drift_check_with_empty_current_sample_reports_not_enough_data(checks_env):
    from hotel_demo.checks import drift

    bookings = pd.read_parquet(data_dir() / "bookings" / "2026-01-11.parquet")
    bookings.iloc[0:0].to_parquet(data_dir() / "bookings" / "2026-02-01.parquet", index=False)

    res = drift.run("2026-02-01")

    _assert_not_enough_data(res, "drift", "2026-02-01")
    assert res["threshold"] == 0.15
    assert res["detail"]["features"] == {feat: None for feat in drift.DRIFT_FEATURES}
    assert res["detail"]["days_evaluated"] == ["2026-02-01"]


def test_drift_check_with_one_empty_feature_uses_max_of_the_others(checks_env):
    from hotel_demo.checks import drift

    bookings = pd.read_parquet(data_dir() / "bookings" / "2026-01-11.parquet")
    bookings["children"] = float("nan")
    bookings["adults"] = 50
    bookings.to_parquet(data_dir() / "bookings" / "2026-02-01.parquet", index=False)

    res = drift.run("2026-02-01")

    features = res["detail"]["features"]
    assert features["children"] is None
    computed = [v for v in features.values() if v is not None]
    assert len(computed) == len(drift.DRIFT_FEATURES) - 1
    assert res["value"] == max(computed)
    assert res["fired"] is True
    assert "not_enough_data" not in res
    assert _read_monitoring_lines("2026-02-01") == [res]


def test_drift_check_without_training_window_reports_not_enough_data(checks_env, monkeypatch):
    from types import SimpleNamespace

    from hotel_demo.checks import drift

    monkeypatch.setattr(
        drift.MlflowClient,
        "get_run",
        lambda self, run_id: SimpleNamespace(data=SimpleNamespace(params={})),
    )

    res = drift.run("2026-01-15")

    _assert_not_enough_data(res, "drift", "2026-01-15")
    assert res["detail"]["days_evaluated"]


def test_running_check_twice_appends_second_line(checks_env):
    from hotel_demo.checks import cancellation_rate

    pred_dir = data_dir() / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "booking_id": [1, 2],
            "business_date": "2026-01-11",
            "model_version": "1",
            "p_cancel": [0.1, 0.2],
        }
    )
    df.to_parquet(pred_dir / "2026-01-11.parquet", index=False)

    start1 = datetime.datetime.now(datetime.UTC)
    res1 = cancellation_rate.run("2026-01-11")
    end1 = datetime.datetime.now(datetime.UTC)

    start2 = datetime.datetime.now(datetime.UTC)
    res2 = cancellation_rate.run("2026-01-11")
    end2 = datetime.datetime.now(datetime.UTC)

    lines = _read_monitoring_lines("2026-01-11")
    assert len(lines) == 2
    assert lines[0] == res1
    assert lines[1] == res2
    _assert_utc_iso_timestamp(lines[0]["ran_at"], start1, end1)
    _assert_utc_iso_timestamp(lines[1]["ran_at"], start2, end2)
    assert lines[0]["ran_at"] <= lines[1]["ran_at"]


def test_write_monitoring_line_stamps_ran_at_and_appends(checks_env):
    from hotel_demo.monitoring import write_monitoring_line

    record = {
        "check": "custom_check",
        "fired": False,
        "value": 0.5,
        "threshold": 1.0,
        "detail": {"info": "test"},
    }
    start = datetime.datetime.now(datetime.UTC)
    res = write_monitoring_line("2026-01-11", record)
    end = datetime.datetime.now(datetime.UTC)

    assert res["check"] == "custom_check"
    assert res["fired"] is False
    assert res["value"] == 0.5
    assert res["threshold"] == 1.0
    assert res["detail"] == {"info": "test"}
    _assert_utc_iso_timestamp(res["ran_at"], start, end)

    lines = _read_monitoring_lines("2026-01-11")
    assert len(lines) == 1
    assert lines[0] == res
