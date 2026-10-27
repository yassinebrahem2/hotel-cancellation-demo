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
```

### Running Tests and Linting

```bash
# Run test suite
uv run pytest

# Lint and format checks
uv run ruff check
uv run ruff format --check
```

## Data Credit & Licence

This project uses data derived from:
- Antonio, N., de Almeida, A. and Nunes, L. (2019). *Hotel booking demand datasets*. **Data in Brief**, 22, 41-49. [doi:10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126).

The source dataset is made available under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence. Note that the original dataset has been re-sliced into daily booking arrival and late outcome files simulating live streaming data in a production environment.
