"""Daily job runner: runs scoring and legacy monitoring checks."""

import datetime
import importlib
import json
import sys
from pathlib import Path

from hotel_demo.paths import data_dir
from hotel_demo.pipeline import score


def daily(date: str) -> None:
    """Run the daily scoring job and all legacy monitoring checks for date.

    Logs execution of each job to data_dir() / "jobs" / f"{date}.jsonl".
    Raises RuntimeError if any step (score or any check) fails.
    """
    jobs_dir = data_dir() / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = jobs_dir / f"{date}.jsonl"

    errors: list[str] = []

    def _log_job(
        job: str,
        started: str,
        ended: str,
        status: str,
        error: str | None,
    ) -> None:
        record = {
            "job": job,
            "business_date": str(date),
            "started": started,
            "ended": ended,
            "status": status,
            "error": error,
        }
        with jobs_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # 1. Run score(date)
    score_started = datetime.datetime.now(datetime.UTC).isoformat()
    try:
        score(date)
        score_ended = datetime.datetime.now(datetime.UTC).isoformat()
        _log_job("score", score_started, score_ended, "success", None)
    except Exception as exc:
        score_ended = datetime.datetime.now(datetime.UTC).isoformat()
        _log_job("score", score_started, score_ended, "failure", str(exc))
        errors.append(f"score: {exc}")

    # 2. Run dynamically discovered checks
    checks_dir = Path(__file__).parent / "checks"
    if checks_dir.exists():
        importlib.invalidate_caches()
        for py_file in sorted(checks_dir.glob("*.py")):
            if py_file.stem == "__init__":
                continue
            job_name = py_file.stem
            chk_started = datetime.datetime.now(datetime.UTC).isoformat()
            try:
                mod_name = f"hotel_demo.checks.{py_file.stem}"
                if mod_name in sys.modules:
                    mod = importlib.reload(sys.modules[mod_name])
                else:
                    mod = importlib.import_module(mod_name)
                job_name = getattr(mod, "CHECK_NAME", py_file.stem)
                if not hasattr(mod, "run") or not callable(mod.run):
                    raise AttributeError(
                        f"Check module {mod_name} does not expose a callable run(date)"
                    )
                mod.run(date)
                chk_ended = datetime.datetime.now(datetime.UTC).isoformat()
                _log_job(job_name, chk_started, chk_ended, "success", None)
            except Exception as exc:
                chk_ended = datetime.datetime.now(datetime.UTC).isoformat()
                _log_job(job_name, chk_started, chk_ended, "failure", str(exc))
                errors.append(f"{job_name}: {exc}")

    if errors:
        raise RuntimeError(f"Daily job failed with {len(errors)} error(s): {'; '.join(errors)}")
