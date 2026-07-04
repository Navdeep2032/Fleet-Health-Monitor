"""
feature_engineering.py

Shared feature computation so training (offline, on CSV) and inference
(online, in dashboard_node.py) use exactly the same features in the same
order. Raw instantaneous readings are noisy; rolling-window statistics over
the last WINDOW samples are what actually separate the fault classes.

Feature vector per timestep (order matters -- must match FEATURE_NAMES):
    current_mean, current_std,
    vibration_mean, vibration_std, vibration_slope,
    temp_mean, temp_std, temp_slope
"""

import numpy as np

WINDOW = 20

FEATURE_NAMES = [
    'current_mean', 'current_std',
    'vibration_mean', 'vibration_std', 'vibration_slope',
    'temp_mean', 'temp_std', 'temp_slope',
]


def _slope(values: np.ndarray) -> float:
    """Simple linear-fit slope over a 1D array of values."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    # polyfit degree 1 -> [slope, intercept]
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def compute_features(current_window, vibration_window, temp_window):
    """
    Given equal-length windows (lists/arrays) of the last WINDOW readings
    for each signal, return the feature vector as a list of floats in
    FEATURE_NAMES order.
    """
    current_window = np.asarray(current_window, dtype=float)
    vibration_window = np.asarray(vibration_window, dtype=float)
    temp_window = np.asarray(temp_window, dtype=float)

    return [
        float(np.mean(current_window)),
        float(np.std(current_window)),
        float(np.mean(vibration_window)),
        float(np.std(vibration_window)),
        _slope(vibration_window),
        float(np.mean(temp_window)),
        float(np.std(temp_window)),
        _slope(temp_window),
    ]
