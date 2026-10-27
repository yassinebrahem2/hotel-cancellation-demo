import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from hotel_demo.features import MODEL_NAME
from hotel_demo.paths import tracking_uri
from hotel_demo.pipeline import train


@pytest.fixture
def temp_environment(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    mlflow_dir = tmp_path / "mlflow"
    data_dir.mkdir()
    mlflow_dir.mkdir()
    (data_dir / "bookings").mkdir()
    (data_dir / "labels").mkdir()

    monkeypatch.setenv("HOTEL_DEMO_DATA", str(data_dir))
    monkeypatch.setenv("HOTEL_DEMO_MLFLOW", str(mlflow_dir))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return {"data": data_dir, "mlflow": mlflow_dir}


def test_uses_labels_booking_changes_and_ignores_late_labels(temp_environment):
    """Verify that train() uses final booking_changes from labels,

    and excludes any booking whose label was released after cutoff,
    and excludes bookings arriving on or after cutoff.
    """
    data = temp_environment["data"]
    b_dir = data / "bookings"
    l_dir = data / "labels"

    # Base booking row generator
    def make_booking(b_id, arr_date, book_changes=0):
        return {
            "booking_id": b_id,
            "booking_date": "2026-01-01",
            "arrival_date": arr_date,
            "hotel": "City Hotel",
            "lead_time": 5,
            "arrival_month": 1,
            "arrival_week": 1,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "booking_changes": book_changes,  # 0 in bookings
            "deposit_type": "No Deposit",
            "agent": 9,
            "company": None,
            "customer_type": "Transient",
            "adr": 100.0,
            "total_of_special_requests": 0,
        }

    # Cutoff is 2026-01-10
    cutoff = "2026-01-10"

    # 1. 20 valid bookings arriving before cutoff (2026-01-05), label released on arrival (2026-01-05 <= cutoff)
    # in bookings: booking_changes = 0; in labels: booking_changes = 5
    valid_bookings = [make_booking(100 + i, "2026-01-05", book_changes=0) for i in range(20)]
    valid_labels = [
        {
            "booking_id": 100 + i,
            "is_canceled": i % 2,
            "label_date": "2026-01-05",
            "booking_changes": 5,  # Non-zero in labels!
        }
        for i in range(20)
    ]

    # 2. Late label booking: arrived before cutoff (2026-01-08), BUT label released after cutoff (2026-01-12)
    late_booking = make_booking(200, "2026-01-08", book_changes=0)
    late_label = {
        "booking_id": 200,
        "is_canceled": 1,
        "label_date": "2026-01-12",  # > cutoff!
        "booking_changes": 3,
    }

    # 3. Late arrival booking: arrived after cutoff (2026-01-15), canceled early (label released 2026-01-02 <= cutoff)
    early_canc_late_arr_booking = make_booking(300, "2026-01-15", book_changes=0)
    early_canc_late_arr_label = {
        "booking_id": 300,
        "is_canceled": 1,
        "label_date": "2026-01-02",  # <= cutoff, but arrival > cutoff!
        "booking_changes": 1,
    }

    all_bookings = [*valid_bookings, late_booking, early_canc_late_arr_booking]
    all_labels = [*valid_labels, late_label, early_canc_late_arr_label]

    pd.DataFrame(all_bookings).to_parquet(b_dir / "2026-01-01.parquet")
    pd.DataFrame(all_labels).to_parquet(l_dir / "2026-01-05.parquet")

    # Run train with cutoff 2026-01-10
    v = train(cutoff=cutoff, min_child_samples=2)

    client = MlflowClient(tracking_uri=tracking_uri())
    mv = client.get_model_version(MODEL_NAME, str(v))
    run = client.get_run(mv.run_id)

    # Only the 20 valid bookings should have been included!
    assert int(run.data.params["total_rows"]) == 20
