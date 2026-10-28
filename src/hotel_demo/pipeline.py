"""Pipeline building, training, evaluation, deployment, and rollback."""

import datetime
import json
import uuid
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.exceptions import MlflowException, RestException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from hotel_demo.features import (
    BOOKING_COLUMNS,
    CATEGORICAL,
    FEATURES,
    MODEL_ALIAS,
    MODEL_NAME,
    TARGET,
    prepare_features,
)
from hotel_demo.paths import data_dir, mlflow_dir, tracking_uri


def build_pipeline(min_child_samples: int | None = None) -> Pipeline:
    """Build the scikit-learn Pipeline with OrdinalEncoder and LightGBM."""
    encoder = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    clf_kwargs: dict[str, Any] = {
        "random_state": 42,
        "verbose": -1,
    }
    if min_child_samples is not None:
        clf_kwargs["min_child_samples"] = min_child_samples

    classifier = LGBMClassifier(**clf_kwargs)

    return Pipeline([("encoder", encoder), ("classifier", classifier)])


def _ensure_experiment(client: MlflowClient) -> str:
    """Ensure the experiment exists with artifact_location under mlflow_dir()/artifacts."""
    exp = client.get_experiment_by_name(MODEL_NAME)
    if exp is None:
        art_path = (mlflow_dir() / "artifacts").resolve()
        art_path.mkdir(parents=True, exist_ok=True)
        exp_id = client.create_experiment(MODEL_NAME, artifact_location=art_path.as_uri())
        return exp_id
    return exp.experiment_id


def _read_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all bookings and labels parquet files from data_dir()."""
    b_dir = data_dir() / "bookings"
    l_dir = data_dir() / "labels"

    booking_files = sorted(b_dir.glob("*.parquet"))
    label_files = sorted(l_dir.glob("*.parquet"))

    if not booking_files:
        raise FileNotFoundError(f"No bookings files found in {b_dir}")
    if not label_files:
        raise FileNotFoundError(f"No labels files found in {l_dir}")

    b_dfs = [pd.read_parquet(f) for f in booking_files]
    l_dfs = [pd.read_parquet(f) for f in label_files]

    bookings = pd.concat(b_dfs, ignore_index=True)
    labels = pd.concat(l_dfs, ignore_index=True)

    return bookings, labels


def train(cutoff: str | datetime.date, min_child_samples: int | None = None) -> str:
    """Train a cancellation model on bookings before cutoff and register it in MLflow."""
    cutoff_date = datetime.date.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff

    bookings, labels = _read_data()

    # Rule: A label released after cutoff is not used
    labels["label_date_dt"] = pd.to_datetime(labels["label_date"]).dt.date
    valid_labels = labels[labels["label_date_dt"] <= cutoff_date].copy()

    # Rule: Keep bookings whose arrival_date is before cutoff
    bookings["arrival_date_dt"] = pd.to_datetime(bookings["arrival_date"]).dt.date
    valid_bookings = bookings[bookings["arrival_date_dt"] < cutoff_date].copy()

    # Take booking_changes from the labels file (the final value)
    valid_bookings = valid_bookings.drop(columns=["booking_changes"])

    # Join bookings and labels on booking_id
    merged = valid_bookings.merge(
        valid_labels[["booking_id", "is_canceled", "label_date", "booking_changes"]],
        on="booking_id",
        how="inner",
    )
    merged = merged.drop_duplicates(subset=["booking_id"], keep="last")

    if len(merged) == 0:
        raise ValueError(f"No training data found before cutoff date {cutoff_date}")

    # Split: 80% train, 20% evaluate (fixed seed 42)
    train_df, eval_df = train_test_split(merged, test_size=0.20, random_state=42)
    eval_ids = [int(x) for x in eval_df["booking_id"]]

    X_train = prepare_features(train_df)
    y_train = train_df[TARGET].astype(int).to_numpy()

    pipeline = build_pipeline(min_child_samples=min_child_samples)
    pipeline.fit(X_train, y_train)

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    _ensure_experiment(client)
    mlflow.set_experiment(MODEL_NAME)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "cutoff": str(cutoff),
                "random_state": 42,
                "min_child_samples": str(min_child_samples)
                if min_child_samples is not None
                else "default",
                "features": ",".join(FEATURES),
                "min_arrival_date": str(train_df["arrival_date_dt"].min()),
                "max_arrival_date": str(train_df["arrival_date_dt"].max()),
                "train_rows": len(train_df),
                "eval_rows": len(eval_df),
                "total_rows": len(merged),
            }
        )

        mlflow.log_dict({"booking_ids": eval_ids}, "eval/booking_ids.json")

        sample_X = X_train.head(5)
        signature = infer_signature(sample_X, pipeline.predict_proba(sample_X))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle",
        )

    version = str(model_info.registered_model_version)
    print(version)
    return version


def evaluate(version: int | str, threshold: float = 0.5) -> dict[str, float]:
    """Score the held-out 20% for the given model version and log metrics to MLflow."""
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    mv = client.get_model_version(MODEL_NAME, str(version))
    art_path = client.download_artifacts(mv.run_id, "eval/booking_ids.json")
    with Path(art_path).open(encoding="utf-8") as f:
        data = json.load(f)
    eval_ids = set(data["booking_ids"])

    bookings, labels = _read_data()
    bookings = bookings.drop(columns=["booking_changes"])
    merged = bookings.merge(
        labels[["booking_id", "is_canceled", "label_date", "booking_changes"]],
        on="booking_id",
        how="inner",
    )
    merged = merged.drop_duplicates(subset=["booking_id"], keep="last")

    eval_df = merged[merged["booking_id"].isin(eval_ids)].copy()
    if len(eval_df) == 0:
        raise ValueError(f"No eval rows found for version {version}")

    X_eval = prepare_features(eval_df)
    y_eval = eval_df[TARGET].astype(int).to_numpy()

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y_eval, y_prob)),
        "roc_auc": float(roc_auc_score(y_eval, y_prob)),
        "precision": float(precision_score(y_eval, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0.0)),
        "brier_score": float(brier_score_loss(y_eval, y_prob)),
    }

    for k, v in metrics.items():
        client.log_metric(mv.run_id, k, v)

    return metrics


def deploy(version: int | str) -> None:
    """Deploy model version to production and record previous version in tags."""
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    target_ver = str(version)
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        prev_ver = str(current.version)
    except (MlflowException, RestException):
        prev_ver = None

    if prev_ver == target_ver:
        return

    if prev_ver is not None:
        client.set_registered_model_tag(MODEL_NAME, "previous_production", prev_ver)

    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, target_ver)


def rollback() -> str:
    """Roll back production to previous version recorded in registered model tags."""
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    rm = client.get_registered_model(MODEL_NAME)
    prev_ver = rm.tags.get("previous_production")
    if not prev_ver:
        raise RuntimeError("No previous production version to roll back to")

    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, prev_ver)
    client.delete_registered_model_tag(MODEL_NAME, "previous_production")
    return prev_ver


def score(date: str) -> Path:
    """Score bookings for the given date using the production model.

    Requires exactly the 27 booking columns in order. Writes predictions atomically
    to data_dir() / "predictions" / f"{date}.parquet".
    """
    b_file = data_dir() / "bookings" / f"{date}.parquet"
    if not b_file.exists():
        raise FileNotFoundError(f"Bookings file not found: {b_file}")

    df = pd.read_parquet(b_file)

    actual_cols = list(df.columns)
    missing = [c for c in BOOKING_COLUMNS if c not in actual_cols]
    unexpected = [c for c in actual_cols if c not in BOOKING_COLUMNS]

    if missing or unexpected:
        raise ValueError(
            f"Schema mismatch in {b_file}: missing columns: {missing}, unexpected columns: {unexpected}"
        )

    if actual_cols != BOOKING_COLUMNS:
        raise ValueError(
            f"Schema column order mismatch in {b_file}: expected {BOOKING_COLUMNS}, got {actual_cols}"
        )

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    try:
        prod_mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except Exception as exc:
        raise RuntimeError(f"No production model found for {MODEL_NAME}@{MODEL_ALIAS}") from exc

    model_version = str(prod_mv.version)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

    X = prepare_features(df)
    p_cancel = model.predict_proba(X)[:, 1]

    pred_df = pd.DataFrame(
        {
            "booking_id": df["booking_id"].astype("int64"),
            "business_date": str(date),
            "model_version": str(model_version),
            "p_cancel": p_cancel.astype("float64"),
        }
    )

    pred_dir = data_dir() / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    target_file = pred_dir / f"{date}.parquet"
    temp_file = pred_dir / f".tmp_{date}_{uuid.uuid4().hex}.parquet"
    try:
        pred_df.to_parquet(temp_file, index=False)
        temp_file.replace(target_file)
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)

    return target_file
