import shlex
import tomllib
from pathlib import Path

from hotel_demo.cli import build_parser


def test_schedule_toml_exists_and_parses():
    schedule_path = Path(__file__).resolve().parent.parent / "ops" / "schedule.toml"
    assert schedule_path.exists(), f"Schedule file not found at {schedule_path}"

    with schedule_path.open("rb") as f:
        data = tomllib.load(f)

    assert "jobs" in data, "schedule.toml missing 'jobs' section"
    assert len(data["jobs"]) >= 1, "schedule.toml has empty 'jobs' list"

    parser = build_parser()

    for job in data["jobs"]:
        assert "name" in job
        assert "command" in job
        assert "every" in job
        assert isinstance(job["every"], int)
        assert job["every"] >= 1

        cmd_str = job["command"].format(date="2026-01-15")
        parts = shlex.split(cmd_str)
        assert parts[0] == "hotel-demo", f"Command must start with hotel-demo, got {parts[0]}"
        # Parse the args following 'hotel-demo'
        parsed = parser.parse_args(parts[1:])
        assert parsed.subcommand is not None
