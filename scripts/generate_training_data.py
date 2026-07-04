"""
generate_training_data.py

Week 3 (ML) data-prep step. Generates many synthetic runs per fault type
using the SAME fault_signatures.generate_sample() function the live ROS
publisher uses, so the offline-trained model matches what it will see live.

Produces data/fleet_training_data.csv with columns:
    run_id, fault_type, cycle, current, vibration, temp

Run from the package root:
    python3 scripts/generate_training_data.py
"""

import csv
import os
import sys

import numpy as np

# allow running this script directly without installing the ROS package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fleet_health_monitor.fault_signatures import generate_sample, FAULT_TYPES

RUNS_PER_FAULT = 40        # independent simulated runs per fault type
CYCLES_PER_RUN = 400       # samples per run
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'fleet_training_data.csv')


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    rows = []
    run_id = 0

    for fault_type in FAULT_TYPES:
        for _ in range(RUNS_PER_FAULT):
            rng = np.random.default_rng(seed=run_id * 97 + 13)
            for cycle in range(CYCLES_PER_RUN):
                current, vibration, temp = generate_sample(cycle, fault_type, rng)
                rows.append([run_id, fault_type, cycle, current, vibration, temp])
            run_id += 1

    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'fault_type', 'cycle', 'current', 'vibration', 'temp'])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows across {run_id} runs to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
