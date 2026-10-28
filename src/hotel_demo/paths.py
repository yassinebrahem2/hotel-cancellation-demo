"""Paths and tracking URI configuration for the hotel demo."""

import os
from pathlib import Path


def data_dir() -> Path:
    """Return the path to the production data folder."""
    env = os.environ.get("HOTEL_DEMO_DATA")
    if env:
        return Path(env)
    return Path.home() / "hotel-demo" / "data"


def mlflow_dir() -> Path:
    """Return the path to the MLflow directory."""
    env = os.environ.get("HOTEL_DEMO_MLFLOW")
    if env:
        return Path(env)
    return Path.home() / "hotel-demo" / "mlflow"


def mlflow_port() -> int:
    """Return the MLflow server port."""
    return int(os.environ.get("HOTEL_DEMO_MLFLOW_PORT", "5000"))


def artifacts_dir() -> Path:
    """Return the path to MLflow artifacts directory."""
    return mlflow_dir() / "artifacts"


def tracking_uri() -> str:
    """Return the MLflow tracking URI.

    Uses MLFLOW_TRACKING_URI if set, otherwise an in-process SQLite URI
    in the MLflow directory.
    """
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    m_dir = mlflow_dir()
    m_dir.mkdir(parents=True, exist_ok=True)
    db_path = (m_dir / "mlflow.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"
