"""
fault_signatures.py

Single source of truth for how each fault type shapes the three telemetry
signals (current, vibration, temperature) as a function of cycle count.

Both the live ROS 2 publisher (unit_telemetry_publisher.py) and the offline
training-data generator (scripts/generate_training_data.py) import this
module, so the simulated signal and the data the model is trained on never
drift apart.

Fault taxonomy:
    HEALTHY        - flat baseline + small noise on all signals
    BEARING_WEAR   - vibration rises and gets noisier; current/temp ~flat
    OVERHEATING    - temperature rises steadily, current creeps up,
                     vibration stays normal
    SENSOR_DROPOUT - one signal randomly flatlines or spikes to a fixed
                     erroneous value
"""

import numpy as np

FAULT_TYPES = ["HEALTHY", "BEARING_WEAR", "OVERHEATING", "SENSOR_DROPOUT"]

# Baseline operating point (arbitrary but consistent units)
BASE_CURRENT = 5.0      # amps
BASE_VIBRATION = 0.20   # g
BASE_TEMP = 40.0        # deg C

NOISE_CURRENT = 0.15
NOISE_VIBRATION = 0.02
NOISE_TEMP = 0.3


def generate_sample(cycle: int, fault_type: str, rng: np.random.Generator):
    """
    Generate one (current, vibration, temp) reading for a given cycle count
    and fault type. `cycle` should increase over the life of a run (e.g. one
    per publisher tick); the further into the run, the more a fault (if any)
    has progressed.

    Returns: (current, vibration, temp) as floats.
    """
    current = BASE_CURRENT + rng.normal(0, NOISE_CURRENT)
    vibration = BASE_VIBRATION + rng.normal(0, NOISE_VIBRATION)
    temp = BASE_TEMP + rng.normal(0, NOISE_TEMP)

    if fault_type == "HEALTHY":
        pass  # signals stay at baseline +/- noise

    elif fault_type == "BEARING_WEAR":
        # Vibration mean AND variance grow with cycle count; current/temp
        # stay essentially flat. Nonlinear growth so it visibly accelerates.
        drift = (cycle / 400.0) ** 1.5
        extra_noise = 1.0 + 4.0 * (cycle / 400.0)
        vibration += drift * 0.6 + rng.normal(0, NOISE_VIBRATION * extra_noise)

    elif fault_type == "OVERHEATING":
        # Temp climbs steadily, current creeps up with it, vibration normal.
        drift = cycle / 300.0
        temp += drift * 25.0
        current += drift * 2.0

    elif fault_type == "SENSOR_DROPOUT":
        # Once past an onset cycle, one signal randomly flatlines or spikes.
        onset = 150
        if cycle > onset and rng.random() < 0.35:
            glitch_signal = rng.choice(["current", "vibration", "temp"])
            if glitch_signal == "current":
                current = 0.0 if rng.random() < 0.5 else BASE_CURRENT * 4
            elif glitch_signal == "vibration":
                vibration = 0.0 if rng.random() < 0.5 else BASE_VIBRATION * 6
            else:
                temp = 0.0 if rng.random() < 0.5 else BASE_TEMP * 2.5
    else:
        raise ValueError(f"Unknown fault_type: {fault_type}")

    return float(current), float(vibration), float(temp)
