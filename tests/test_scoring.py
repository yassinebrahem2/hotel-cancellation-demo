import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from hotel_demo.features import MODEL_ALIAS, MODEL_NAME
from hotel_demo.paths import data_dir, tracking_uri
from hotel_demo.pipeline import deploy, score, train


@pytest.fixture
def scoring_env(tmp_path, monkeypatch):
    """Set up an isolated data and MLflow environment with a trained and deployed model."""
    data_target = tmp_path / "data"
    mlflow_target = tmp_path / "mlflow"
    fixture_data = Path(__file__).parent / "fixtures" / "data"

    shutil.copytree(fixture_data, data_target)
    mlflow_target.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOTEL_DEMO_DATA", str(data_target))
    monkeypatch.setenv("HOTEL_DEMO_MLFLOW", str(mlflow_target))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    # Train and deploy version 1
    v1 = train(cutoff="2026-01-10", min_child_samples=5)
    deploy(version=v1)

    return {"data": data_target, "mlflow": mlflow_target, "version": v1}


def test_score_produces_valid_predictions(scoring_env):
    pred_path = score("2026-01-11")
    assert pred_path.exists()
    assert pred_path == data_dir() / "predictions" / "2026-01-11.parquet"

    df_preds = pd.read_parquet(pred_path)
    assert list(df_preds.columns) == ["booking_id", "business_date", "model_version", "p_cancel"]
    assert df_preds["booking_id"].dtype == np.int64
    assert df_preds["p_cancel"].dtype == np.float64
    assert (df_preds["business_date"] == "2026-01-11").all()
    assert (df_preds["model_version"] == scoring_env["version"]).all()
    assert len(df_preds) == 12
    assert (df_preds["p_cancel"] >= 0.0).all() and (df_preds["p_cancel"] <= 1.0).all()


def test_score_missing_file_raises(scoring_env):
    with pytest.raises(FileNotFoundError, match=r"2099-01-01\.parquet"):
        score("2099-01-01")


def test_score_missing_column_fails_and_names_column(scoring_env):
    b_file = data_dir() / "bookings" / "2026-01-11.parquet"
    df = pd.read_parquet(b_file)
    df_missing = df.drop(columns=["adults"])
    bad_file = data_dir() / "bookings" / "bad_missing.parquet"
    df_missing.to_parquet(bad_file, index=False)

    with pytest.raises(ValueError) as excinfo:
        score("bad_missing")
    err_msg = str(excinfo.value)
    assert "bad_missing.parquet" in err_msg
    assert "adults" in err_msg
    # Ensure no predictions file was written
    assert not (data_dir() / "predictions" / "bad_missing.parquet").exists()


def test_score_unexpected_column_fails_and_names_column(scoring_env):
    b_file = data_dir() / "bookings" / "2026-01-11.parquet"
    df = pd.read_parquet(b_file)
    df_extra = df.copy()
    df_extra["surprise_feature"] = 123
    bad_file = data_dir() / "bookings" / "bad_extra.parquet"
    df_extra.to_parquet(bad_file, index=False)

    with pytest.raises(ValueError) as excinfo:
        score("bad_extra")
    err_msg = str(excinfo.value)
    assert "bad_extra.parquet" in err_msg
    assert "surprise_feature" in err_msg
    assert not (data_dir() / "predictions" / "bad_extra.parquet").exists()


def test_score_column_order_mismatch_fails(scoring_env):
    b_file = data_dir() / "bookings" / "2026-01-11.parquet"
    df = pd.read_parquet(b_file)
    # Swap two columns
    cols = list(df.columns)
    cols[0], cols[1] = cols[1], cols[0]
    bad_file = data_dir() / "bookings" / "bad_order.parquet"
    df[cols].to_parquet(bad_file, index=False)

    with pytest.raises(ValueError) as excinfo:
        score("bad_order")
    assert "bad_order.parquet" in str(excinfo.value)
    assert not (data_dir() / "predictions" / "bad_order.parquet").exists()


def test_score_no_production_model_raises(scoring_env):
    client = MlflowClient(tracking_uri=tracking_uri())
    client.delete_registered_model_alias(MODEL_NAME, MODEL_ALIAS)

    with pytest.raises(Exception, match=r"[Pp]roduction"):
        score("2026-01-11")


def test_score_unseen_country_or_agent(scoring_env):
    b_file = data_dir() / "bookings" / "2026-01-11.parquet"
    df = pd.read_parquet(b_file)
    df.loc[0, "country"] = "ZZZ_UNSEEN_LAND"
    df.loc[0, "agent"] = 9999999
    unseen_file = data_dir() / "bookings" / "unseen.parquet"
    df.to_parquet(unseen_file, index=False)

    pred_path = score("unseen")
    assert pred_path.exists()
    preds = pd.read_parquet(pred_path)
    assert not np.isnan(preds.loc[0, "p_cancel"])


def test_score_null_and_non_null_variants(scoring_env):
    b_file = data_dir() / "bookings" / "2026-01-11.parquet"
    df = pd.read_parquet(b_file)

    # Variant A: All nulls in children, agent, company
    df_nulls = df.copy()
    df_nulls["children"] = None
    df_nulls["agent"] = None
    df_nulls["company"] = None
    f_nulls = data_dir() / "bookings" / "nulls.parquet"
    df_nulls.to_parquet(f_nulls, index=False)

    # Variant B: No nulls in children, agent, company
    df_nonulls = df.copy()
    df_nonulls["children"] = 2
    df_nonulls["agent"] = 15
    df_nonulls["company"] = 20
    f_nonulls = data_dir() / "bookings" / "nonulls.parquet"
    df_nonulls.to_parquet(f_nonulls, index=False)

    p_nulls = score("nulls")
    p_nonulls = score("nonulls")

    assert p_nulls.exists()
    assert p_nonulls.exists()

    res_nulls = pd.read_parquet(p_nulls)
    res_nonulls = pd.read_parquet(p_nonulls)

    assert not res_nulls["p_cancel"].isna().any()
    assert not res_nonulls["p_cancel"].isna().any()
