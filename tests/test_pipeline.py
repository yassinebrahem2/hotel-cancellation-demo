from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from hotel_demo.features import (
    FEATURES,
    MODEL_ALIAS,
    MODEL_NAME,
    prepare_features,
)
from hotel_demo.paths import data_dir, mlflow_dir, tracking_uri
from hotel_demo.pipeline import deploy, evaluate, rollback, train


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Set up isolated MLflow and data directories for testing."""
    mlflow_target = tmp_path / "mlflow"
    mlflow_target.mkdir(parents=True, exist_ok=True)
    fixture_data = Path(__file__).parent / "fixtures" / "data"

    monkeypatch.setenv("HOTEL_DEMO_DATA", str(fixture_data))
    monkeypatch.setenv("HOTEL_DEMO_MLFLOW", str(mlflow_target))
    # Clear any global tracking URI in env
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    return {"mlflow": mlflow_target, "data": fixture_data}


def test_paths_default_and_override(tmp_path, monkeypatch):
    monkeypatch.delenv("HOTEL_DEMO_DATA", raising=False)
    monkeypatch.delenv("HOTEL_DEMO_MLFLOW", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert data_dir() == Path.home() / "hotel-demo" / "data"
    assert mlflow_dir() == Path.home() / "hotel-demo" / "mlflow"
    assert tracking_uri().startswith("sqlite:///")

    custom_data = tmp_path / "custom_data"
    custom_mlflow = tmp_path / "custom_mlflow"
    monkeypatch.setenv("HOTEL_DEMO_DATA", str(custom_data))
    monkeypatch.setenv("HOTEL_DEMO_MLFLOW", str(custom_mlflow))

    assert data_dir() == custom_data
    assert mlflow_dir() == custom_mlflow
    assert str(custom_mlflow.as_posix()) in tracking_uri()

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    assert tracking_uri() == "http://localhost:5000"


def test_prepare_features():
    df = pd.DataFrame(
        {
            "booking_id": [1, 2],
            "booking_date": ["2026-01-01", "2026-01-02"],
            "arrival_date": ["2026-01-10", "2026-01-11"],
            "hotel": ["City Hotel", "Resort Hotel"],
            "lead_time": [9, 9],
            "arrival_month": [1, 1],
            "arrival_week": [2, 2],
            "stays_in_weekend_nights": [0, 1],
            "stays_in_week_nights": [2, 3],
            "adults": [2, 1],
            "children": [0, None],  # integer and missing
            "babies": [0, 0],
            "meal": ["BB", None],
            "country": ["PRT", None],
            "market_segment": ["Online TA", "Direct"],
            "distribution_channel": ["TA/TO", "Direct"],
            "is_repeated_guest": [0, 0],
            "previous_cancellations": [0, 0],
            "previous_bookings_not_canceled": [0, 0],
            "reserved_room_type": ["A", "D"],
            "booking_changes": [1, 0],
            "deposit_type": ["No Deposit", "No Deposit"],
            "agent": [9, None],  # ID as int and missing
            "company": [None, 40],  # missing and ID
            "customer_type": ["Transient", "Transient"],
            "adr": [100.0, 120.0],
            "total_of_special_requests": [1, 0],
            "extra_col": ["foo", "bar"],
        }
    )

    prepared = prepare_features(df)
    assert list(prepared.columns) == FEATURES
    assert "booking_id" not in prepared.columns
    assert "extra_col" not in prepared.columns
    assert prepared["children"].dtype == np.float64
    assert pd.isna(prepared.loc[1, "children"])
    assert prepared.loc[0, "agent"] == "9"
    assert pd.isna(prepared.loc[1, "agent"])
    assert pd.isna(prepared.loc[0, "company"])
    assert prepared.loc[1, "company"] == "40"


def test_train_stop_conditions(isolated_env, tmp_path, monkeypatch):
    # Run from another directory to test that no mlruns is created in cwd
    work_dir = tmp_path / "other_work_dir"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    version = train(cutoff="2026-01-10", min_child_samples=5)
    assert str(version) == "1"

    client = MlflowClient(tracking_uri=tracking_uri())

    # 1. Production alias must NOT be set
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

    # 2. Check registered model version 1 exists
    mv = client.get_model_version(MODEL_NAME, "1")
    assert str(mv.version) == "1"
    run_id = mv.run_id

    # 3. Check logged params
    run = client.get_run(run_id)
    params = run.data.params
    assert params["cutoff"] == "2026-01-10"
    assert params["min_child_samples"] == "5"
    assert "features" in params
    assert "min_arrival_date" in params
    assert "max_arrival_date" in params
    assert "train_rows" in params
    assert "eval_rows" in params

    # 4. Held out IDs logged as artifact
    artifacts = [a.path for a in client.list_artifacts(run_id, "eval")]
    assert "eval/booking_ids.json" in artifacts

    # 5. Check no mlruns created in working directory
    assert not (work_dir / "mlruns").exists()


def test_evaluate_stop_conditions(isolated_env):
    version = train(cutoff="2026-01-10", min_child_samples=5)
    metrics = evaluate(version=version, threshold=0.5)

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "brier_score" in metrics

    # Verify they are logged to MLflow run
    client = MlflowClient(tracking_uri=tracking_uri())
    mv = client.get_model_version(MODEL_NAME, str(version))
    run = client.get_run(mv.run_id)
    logged_metrics = run.data.metrics

    for k in ["pr_auc", "roc_auc", "precision", "recall", "brier_score"]:
        assert k in logged_metrics


def test_deploy_and_rollback(isolated_env):
    # Train version 1
    v1 = train(cutoff="2026-01-10", min_child_samples=5)
    client = MlflowClient(tracking_uri=tracking_uri())

    # Deploy version 1
    deploy(version=v1)
    prod = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    assert str(prod.version) == str(v1)

    # Train version 2
    v2 = train(cutoff="2026-01-10", min_child_samples=5)
    assert str(v2) == "2"

    # Deploy version 2
    deploy(version=v2)
    prod = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    assert str(prod.version) == str(v2)

    # Rollback to version 1
    rb_ver = rollback()
    assert str(rb_ver) == str(v1)
    prod = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    assert str(prod.version) == str(v1)

    # Calling rollback again fails clearly
    with pytest.raises(RuntimeError, match=r"[Nn]o previous"):
        rollback()


def test_rollback_without_prior_deploy_fails(isolated_env):
    train(cutoff="2026-01-10", min_child_samples=5)
    with pytest.raises(RuntimeError, match=r"[Nn]o previous"):
        rollback()


def test_deploy_already_live_version_changes_nothing(isolated_env):
    v1 = train(cutoff="2026-01-10", min_child_samples=5)
    deploy(version=v1)
    client = MlflowClient(tracking_uri=tracking_uri())
    rm = client.get_registered_model(MODEL_NAME)
    assert "previous_production" not in rm.tags

    v2 = train(cutoff="2026-01-10", min_child_samples=5)
    deploy(version=v2)
    rm = client.get_registered_model(MODEL_NAME)
    assert rm.tags["previous_production"] == str(v1)

    # Deploying v2 again changes nothing: does not overwrite previous_production tag
    deploy(version=v2)
    rm = client.get_registered_model(MODEL_NAME)
    assert rm.tags["previous_production"] == str(v1)

    # Rollback still successfully rolls back to v1
    assert rollback() == str(v1)
