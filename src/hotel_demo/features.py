"""Features, targets, and input preparation for the hotel cancellation model."""

import pandas as pd

TARGET = "is_canceled"

FEATURES = [
    "hotel",
    "lead_time",
    "arrival_month",
    "arrival_week",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "customer_type",
    "adr",
    "total_of_special_requests",
]

CATEGORICAL = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent",
    "company",
]

MODEL_NAME = "hotel-cancellation"
MODEL_ALIAS = "production"


def _format_id(val: object) -> object:
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(val)
    return str(val)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the model input features DataFrame for both train and score.

    Ensures children is float64, agent and company are strings with nulls
    preserved, and returns only FEATURES.
    """
    out = df[FEATURES].copy()
    out["children"] = out["children"].astype("float64")
    out["agent"] = out["agent"].apply(_format_id)
    out["company"] = out["company"].apply(_format_id)

    # Ensure other categorical string columns keep nulls and avoid categorical dtype
    for col in CATEGORICAL:
        if col not in ("agent", "company"):
            out[col] = out[col].apply(lambda x: None if (pd.isna(x) or x is None) else str(x))

    return out
