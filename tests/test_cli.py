import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hotel_demo.cli import main


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    mlflow_target = tmp_path / "mlflow"
    mlflow_target.mkdir(parents=True, exist_ok=True)
    fixture_data = Path(__file__).parent / "fixtures" / "data"

    monkeypatch.setenv("HOTEL_DEMO_DATA", str(fixture_data))
    monkeypatch.setenv("HOTEL_DEMO_MLFLOW", str(mlflow_target))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return {"mlflow": mlflow_target, "data": fixture_data}


def test_cli_train_evaluate_deploy_rollback(cli_env, capsys):
    # Train
    with patch.object(
        sys, "argv", ["hotel-demo", "train", "--cutoff", "2026-01-10", "--min-child-samples", "5"]
    ):
        ret = main()
        assert ret == 0
        captured = capsys.readouterr()
        lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
        assert lines[-1] == "1"

    # Evaluate
    with patch.object(
        sys, "argv", ["hotel-demo", "evaluate", "--version", "1", "--threshold", "0.5"]
    ):
        ret = main()
        assert ret == 0
        captured = capsys.readouterr()
        assert "pr_auc" in captured.out

    # Deploy
    with patch.object(sys, "argv", ["hotel-demo", "deploy", "--version", "1"]):
        ret = main()
        assert ret == 0

    # Train v2
    with patch.object(
        sys, "argv", ["hotel-demo", "train", "--cutoff", "2026-01-10", "--min-child-samples", "5"]
    ):
        ret = main()
        assert ret == 0
        captured = capsys.readouterr()
        lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
        assert lines[-1] == "2"

    # Deploy v2
    with patch.object(sys, "argv", ["hotel-demo", "deploy", "--version", "2"]):
        ret = main()
        assert ret == 0

    # Rollback (rolls back to 1)
    with patch.object(sys, "argv", ["hotel-demo", "rollback"]):
        ret = main()
        assert ret == 0

    # Rollback again (fails because no previous version)
    with patch.object(sys, "argv", ["hotel-demo", "rollback"]):
        ret = main()
        assert ret != 0
        captured = capsys.readouterr()
        assert "No previous" in captured.err or "No previous" in captured.out


def test_cli_failure_cases(cli_env, capsys):
    # Missing required argument for train
    with patch.object(sys, "argv", ["hotel-demo", "train"]):
        ret = main()
        assert ret != 0

    # Unknown command
    with patch.object(sys, "argv", ["hotel-demo", "invalid-cmd"]):
        ret = main()
        assert ret != 0
