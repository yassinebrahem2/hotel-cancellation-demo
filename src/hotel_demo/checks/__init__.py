"""Legacy monitoring checks for the hotel cancellation demo."""

from hotel_demo.checks import accuracy, cancellation_rate, drift

__all__ = ["accuracy", "cancellation_rate", "drift"]
