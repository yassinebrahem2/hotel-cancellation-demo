import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from hotel_demo.cli import main
from hotel_demo.daily import daily
from hotel_demo.paths import data_dir
from hotel_demo.pipeline import deploy, train


@pytest.fixture
def daily_env(tmp_path, monkeypatch):
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


def test_daily_writes_predictions_monitoring_and_jobs(daily_env):
    daily("2026-01-11")

    # 1. Predictions file written
    pred_file = data_dir() / "predictions" / "2026-01-11.parquet"
    assert pred_file.exists()
    df_preds = pd.read_parquet(pred_file)
    assert len(df_preds) > 0

    # 2. Monitoring file written with 3 check lines
    mon_file = data_dir() / "monitoring" / "2026-01-11.jsonl"
    assert mon_file.exists()
    mon_lines = []
    with mon_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                mon_lines.append(json.loads(line))
    assert len(mon_lines) == 3
    checks_run = {line["check"] for line in mon_lines}
    assert checks_run == {"cancellation_rate", "drift", "accuracy"}

    # 3. Jobs file written with jobs: score and the checks
    jobs_file = data_dir() / "jobs" / "2026-01-11.jsonl"
    assert jobs_file.exists()
    job_lines = []
    with jobs_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                job_lines.append(json.loads(line))
    assert len(job_lines) >= 4  # score + 3 checks
    for j in job_lines:
        assert set(j.keys()) == {"job", "business_date", "started", "ended", "status", "error"}
        assert j["business_date"] == "2026-01-11"
        assert j["status"] == "success"
        assert j["error"] is None


def test_daily_dynamic_check_discovery(daily_env):
    # Add a custom check to src/hotel_demo/checks
    checks_dir = Path(__file__).resolve().parent.parent / "src" / "hotel_demo" / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    custom_check_path = checks_dir / "custom_test_check.py"

    check_code = """
from typing import Any
import json
from hotel_demo.paths import data_dir

CHECK_NAME = "custom_test_check"

def run(date: str) -> dict[str, Any]:
    record = {
        "check": CHECK_NAME,
        "fired": False,
        "value": 42.0,
        "threshold": 100.0,
        "detail": {"hello": "world"},
    }
    mon_file = data_dir() / "monitoring" / f"{date}.jsonl"
    with mon_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\\n")
    return record
"""
    custom_check_path.write_text(check_code, encoding="utf-8")

    try:
        daily("2026-01-11")

        mon_file = data_dir() / "monitoring" / "2026-01-11.jsonl"
        mon_lines = []
        with mon_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    mon_lines.append(json.loads(line))

        checks_run = {line["check"] for line in mon_lines}
        assert "custom_test_check" in checks_run

        jobs_file = data_dir() / "jobs" / "2026-01-11.jsonl"
        with jobs_file.open("r", encoding="utf-8") as f:
            jobs_run = [json.loads(line)["job"] for line in f if line.strip()]
        assert "custom_test_check" in jobs_run

    finally:
        if custom_check_path.exists():
            custom_check_path.unlink()


def test_daily_check_file_import_error_logs_failed_job_and_runs_others(daily_env):
    checks_dir = Path(__file__).parent.parent / "src" / "hotel_demo" / "checks"
    bad_check_path = checks_dir / "bad_syntax_check.py"
    bad_check_path.write_text("invalid python syntax :::", encoding="utf-8")

    try:
        with pytest.raises(RuntimeError) as excinfo:
            daily("2026-01-11")
        assert "bad_syntax_check" in str(excinfo.value)

        # Check that jobs file logged bad_syntax_check as failure
        jobs_file = data_dir() / "jobs" / "2026-01-11.jsonl"
        with jobs_file.open("r", encoding="utf-8") as f:
            jobs = [json.loads(line) for line in f if line.strip()]

        bad_job = next(j for j in jobs if j["job"] == "bad_syntax_check")
        assert bad_job["status"] == "failure"
        assert bad_job["error"] is not None

        # Other checks (e.g. drift) still ran and logged to jobs
        drift_job = next(j for j in jobs if j["job"] == "drift")
        assert drift_job["status"] == "success"
    finally:
        if bad_check_path.exists():
            bad_check_path.unlink()


def test_daily_score_failure_still_runs_other_checks(daily_env):
    # Corrupt bookings file for 2026-01-15 with extra column
    b_file = data_dir() / "bookings" / "2026-01-15.parquet"
    df = pd.read_parquet(b_file)
    df["extra_bad_col"] = 999
    df.to_parquet(b_file, index=False)

    # daily should raise or return non-zero
    with pytest.raises(RuntimeError) as excinfo:
        daily("2026-01-15")
    assert "extra_bad_col" in str(excinfo.value)

    # No predictions written for 2026-01-15
    assert not (data_dir() / "predictions" / "2026-01-15.parquet").exists()

    # Jobs file logs score failure
    jobs_file = data_dir() / "jobs" / "2026-01-15.jsonl"
    assert jobs_file.exists()
    with jobs_file.open("r", encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f if line.strip()]

    score_job = next(j for j in jobs if j["job"] == "score")
    assert score_job["status"] == "failure"
    assert "extra_bad_col" in score_job["error"]

    # Other checks that do not need today's predictions (e.g. drift) still ran!
    drift_job = next(j for j in jobs if j["job"] == "drift")
    assert drift_job["status"] == "success"

    mon_file = data_dir() / "monitoring" / "2026-01-15.jsonl"
    assert mon_file.exists()
    with mon_file.open("r", encoding="utf-8") as f:
        mon_checks = [json.loads(line)["check"] for line in f if line.strip()]
    assert "drift" in mon_checks


def test_cli_score_and_daily(daily_env):
    # Score command via CLI
    code = main(["score", "--date", "2026-01-11"])
    assert code == 0
    assert (data_dir() / "predictions" / "2026-01-11.parquet").exists()

    # Daily command via CLI
    code = main(["daily", "--date", "2026-01-12"])
    assert code == 0
    assert (data_dir() / "predictions" / "2026-01-12.parquet").exists()
    assert (data_dir() / "monitoring" / "2026-01-12.jsonl").exists()
    assert (data_dir() / "jobs" / "2026-01-12.jsonl").exists()

    # Failure via CLI returns non-zero
    code = main(["score", "--date", "2099-99-99"])
    assert code != 0
