"""CLI entry point for the hotel demo."""

import argparse
import sys
from collections.abc import Sequence

from hotel_demo.daily import daily
from hotel_demo.pipeline import deploy, evaluate, rollback, score, train


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for hotel-demo subcommands."""
    parser = argparse.ArgumentParser(
        prog="hotel-demo",
        description="Hotel cancellation model operations",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Train a model version up to cutoff date")
    train_parser.add_argument(
        "--cutoff",
        required=True,
        help="Cutoff date YYYY-MM-DD (bookings with arrival_date before this are used)",
    )
    train_parser.add_argument(
        "--min-child-samples",
        type=int,
        default=None,
        help="Minimum number of data needed in a leaf",
    )

    # evaluate
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a model version on held-out data"
    )
    eval_parser.add_argument("--version", required=True, help="Model version to evaluate")
    eval_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification (default 0.5)",
    )

    # deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a model version to production")
    deploy_parser.add_argument("--version", required=True, help="Model version to deploy")

    # rollback
    subparsers.add_parser("rollback", help="Roll back production to previous version")

    # score
    score_parser = subparsers.add_parser("score", help="Score bookings for a given date")
    score_parser.add_argument("--date", required=True, help="Business date YYYY-MM-DD")

    # daily
    daily_parser = subparsers.add_parser(
        "daily", help="Run daily scoring and monitoring checks for a given date"
    )
    daily_parser.add_argument("--date", required=True, help="Business date YYYY-MM-DD")

    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Run CLI subcommands. Returns exit code (0 on success, non-zero on failure)."""
    parser = build_parser()
    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code

    try:
        if parsed.subcommand == "train":
            train(cutoff=parsed.cutoff, min_child_samples=parsed.min_child_samples)
            return 0
        elif parsed.subcommand == "evaluate":
            metrics = evaluate(version=parsed.version, threshold=parsed.threshold)
            print(f"Evaluation metrics for version {parsed.version}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            return 0
        elif parsed.subcommand == "deploy":
            deploy(version=parsed.version)
            print(f"Deployed version {parsed.version} to production")
            return 0
        elif parsed.subcommand == "rollback":
            prev_ver = rollback()
            print(f"Rolled back production to version {prev_ver}")
            return 0
        elif parsed.subcommand == "score":
            score(date=parsed.date)
            print(f"Scored bookings for {parsed.date}")
            return 0
        elif parsed.subcommand == "daily":
            daily(date=parsed.date)
            print(f"Daily job completed for {parsed.date}")
            return 0
        else:
            sys.stderr.write(f"Unknown subcommand: {parsed.subcommand}\n")
            return 1
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
