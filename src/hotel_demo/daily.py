"""Daily job runner: runs scoring and legacy monitoring checks."""

import datetime
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from hotel_demo.paths import data_dir
from hotel_demo.pipeline import score

CHECKS_DIR = Path(__file__).parent / "checks"


def _load_check(py_file: Path) -> ModuleType:
    """Execute py_file as a fresh module named hotel_demo.checks.<stem>."""
    mod_name = f"hotel_demo.checks.{py_file.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load check module {mod_name} from {py_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def daily(date: str, *, checks_dir: Path | None = None) -> None:
    """Run the daily scoring job and all legacy monitoring checks for date.

    Checks are discovered from checks_dir (default: the package's checks directory).
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
    discovery_dir = CHECKS_DIR if checks_dir is None else checks_dir
    if discovery_dir.exists():
        for py_file in sorted(discovery_dir.glob("*.py")):
            if py_file.stem == "__init__":
                continue
            job_name = py_file.stem
            chk_started = datetime.datetime.now(datetime.UTC).isoformat()
            try:
                mod = _load_check(py_file)
                job_name = getattr(mod, "CHECK_NAME", py_file.stem)
                if not hasattr(mod, "run") or not callable(mod.run):
                    raise AttributeError(
                        f"Check module {mod.__name__} does not expose a callable run(date)"
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
