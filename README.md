# Hotel Cancellation Demo

Production machine learning system for hotel booking cancellation prediction.

## System Overview

The hotel group's machine learning system predicts the cancellation probability of each reservation at booking time. The revenue management team aggregates the expected cancellation counts per arrival date to guide overbooking decisions and maximize occupancy without turning away guests who hold confirmed reservations.

The system features:
- **Pipeline**: Scikit-learn `Pipeline` combining categorical encoding (`OrdinalEncoder` handling unseen categories) with a `LightGBM` classifier.
- **Model Registry & Tracking**: Logged and registered in MLflow under the model name `hotel-cancellation`.
- **Deployment Management**: Release management via the `production` model alias with tracking of the previous production version to enable safe rollback.
- **Reproducible Evaluation**: Held-out evaluation sets are saved as run artifacts ensuring consistent evaluation of model versions.

## Getting Started

### Installation

Install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### Configuration

The system uses environment variables for path resolution (falling back to `~/hotel-demo/`):
- `HOTEL_DEMO_DATA`: Path to production data folder containing `bookings/` and `labels/` parquet files.
- `HOTEL_DEMO_MLFLOW`: Path to the MLflow directory where SQLite metadata (`mlflow.db`) and artifacts reside.
- `MLFLOW_TRACKING_URI`: Tracking URI override (defaults to `sqlite:///<HOTEL_DEMO_MLFLOW>/mlflow.db`).

### CLI Usage

The package provides the `hotel-demo` command line tool:

```bash
# Train a model on all historical data arriving before the cutoff date
uv run hotel-demo train --cutoff 2026-01-10 [--min-child-samples N]

# Evaluate a specific model version on its held-out test split
uv run hotel-demo evaluate --version 1 [--threshold 0.5]

# Deploy a model version to the production alias
uv run hotel-demo deploy --version 1

# Roll back production alias to the previous production version
uv run hotel-demo rollback

# Score incoming bookings for a specific date
uv run hotel-demo score --date 2026-01-11

# Run daily scoring and all legacy monitoring checks
uv run hotel-demo daily --date 2026-01-11
```

### Running Tests and Linting

```bash
# Run test suite
uv run pytest

# Lint and format checks
uv run ruff check
uv run ruff format --check
```

## Production Schedule & CI Workflows

### Production Schedule (`ops/schedule.toml`)

The company's production schedule is defined in `ops/schedule.toml`. On the company's ops host, a scheduler runs the jobs configured in this file (such as `hotel-demo daily --date {date}`) on each business day. Model retraining and deployment are not automated on a schedule; engineers and data scientists execute `train`, `evaluate`, and `deploy` manually.

### CI Smoke Run (`.github/workflows/scheduled.yml`)

The GitHub Actions workflow `.github/workflows/scheduled.yml` serves as a CI smoke run executed on a daily schedule and manually via `workflow_dispatch`. It exercises the end-to-end ML lifecycle on a small slice of sample data: downloading the data archive, training a model version, evaluating performance metrics, deploying to `production`, and executing the daily batch job.

This workflow is strictly a CI smoke test and is distinct from the company's production schedule defined in `ops/schedule.toml`.

#### Configuring Sample Data URL (`SAMPLE_DATA_URL`)

By default, the scheduled smoke run downloads `ops/sample-data.tar.gz` from the repository. To supply an alternate or external sample dataset location, set the `SAMPLE_DATA_URL` repository variable:

1. Navigate to your repository on GitHub.
2. Go to **Settings** > **Secrets and variables** > **Actions**.
3. Under the **Variables** tab, click **New repository variable**.
4. Set **Name** to `SAMPLE_DATA_URL` and **Value** to the direct URL of the `sample-data.tar.gz` archive.

## Data Credit & Licence

This project uses data derived from:
- Antonio, N., de Almeida, A. and Nunes, L. (2019). *Hotel booking demand datasets*. **Data in Brief**, 22, 41-49. [doi:10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126).

The source dataset is made available under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence. Note that the original dataset has been re-sliced into daily booking arrival and late outcome files simulating live streaming data in a production environment.
