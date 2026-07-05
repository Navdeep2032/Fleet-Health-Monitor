# Fleet Health Monitor — Multi-Node Telemetry + ML Fault Classification

**Weekend Task 4 — Software Final Capstone**

## Which weeks this bridges

| Week                          | What it contributes                                                                                                                                                                                               |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Week 1 — ROS 2 plumbing**   | 4 independent, namespaced publisher nodes (`/unit_1/telemetry` … `/unit_4/telemetry`), a central subscriber/aggregator node (`dashboard_node`), and a logger node — all brought up together from one launch file. |
| **Week 3 — Machine learning** | An offline-trained `RandomForestClassifier` that classifies each unit's rolling telemetry window into one of 4 fault classes, run live inside the dashboard node.                                                 |

This project does not use Week 2 (Nav2/Gazebo) — it's a 2-way fusion by design, per the "at least two" requirement.

## What this system can do

- Run 4 independent, namespaced ROS 2 nodes simultaneously, each simulating
  one fleet unit's telemetry (`current`, `vibration`, `temp`) at 10 Hz.
- Classify each unit's live rolling window into one of 4 states —
  `HEALTHY`, `BEARING_WEAR`, `OVERHEATING`, `SENSOR_DROPOUT` — using a
  pre-trained `RandomForestClassifier`, updated continuously as new data
  arrives.
- Detect gradually-developing faults via trend features (`vibration_std`,
  `temp_slope`, etc.), not just instantaneous threshold crossings.
- Report live status via a color-coded terminal table, a `/fleet/status`
  ROS topic, a CSV mission log, and a standalone Tkinter GUI window.
- Log only on state _transitions_ (not every message) to avoid terminal
  flooding, while still catching every fault onset.
- Bring the entire fleet + dashboard + logger up with a single launch file.
- Retrain on new fault types without any architectural change — just add
  a new signature to `fault_signatures.py`, regenerate training data, and
  retrain.

## Architecture

```
 unit_1 (HEALTHY)      \
 unit_2 (BEARING_WEAR)  \
 unit_3 (OVERHEATING)    >----  /unit_N/telemetry  ---->  dashboard_node  ---->  /fleet/status
 unit_4 (HEALTHY)       /       (Float32MultiArray:            |                      |
                        /        [current, vibration,          | loads                v
                                  temp, cycle], 10 Hz)          | fault_classifier.pkl  telemetry_logger
                                                                 | rolling-window        (CSV mission log)
                                                                 | feature calc +
                                                                 | live .predict()
```

Each `unit_telemetry_publisher` node simulates one motor/subsystem publishing
`[current, vibration, temp, cycle]` as a `Float32MultiArray` at 10 Hz. Each
is launched with its own `unit_id` and `fault_type` parameter — same code,
different config, which is the "multi-node plumbing" idea from Week 1.

`dashboard_node` subscribes to all 4 topics, keeps a 20-sample rolling
buffer per unit, computes 8 features (mean/std of current, mean/std/slope
of vibration, mean/std/slope of temp), and runs the pre-trained classifier
on each unit independently. It prints a live color-coded table and
publishes a summary string to `/fleet/status` in the format
`unit_id:label:confidence:cycle` (pipe-separated across units), which both
`telemetry_logger.py` and `gui_monitor.py` consume.

## Fault taxonomy

| Class            | Signature                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `HEALTHY`        | All 3 signals flat at baseline ± small noise                                                                             |
| `BEARING_WEAR`   | Vibration mean _and_ variance grow nonlinearly over time; current/temp stay flat                                         |
| `OVERHEATING`    | Temperature climbs steadily, current creeps up with it, vibration normal                                                 |
| `SENSOR_DROPOUT` | After an onset cycle, one signal randomly flatlines or spikes to a fixed bad value (intermittent, ~35% chance per cycle) |

Signal generation lives in one shared module
(`fleet_health_monitor/fault_signatures.py`) used by **both** the live ROS
publisher and the offline training-data generator, so the model is trained
on exactly the distribution it sees at inference time.

## Model performance (held-out test set, 25% split)

| Class          | Precision | Recall | F1   |
| -------------- | --------- | ------ | ---- |
| BEARING_WEAR   | 1.00      | 0.96   | 0.98 |
| HEALTHY        | 0.72      | 1.00   | 0.83 |
| OVERHEATING    | 1.00      | 1.00   | 1.00 |
| SENSOR_DROPOUT | 1.00      | 0.64   | 0.78 |

**Overall accuracy: 90%.**

## Real run evidence

The demo below is from an actual `ros2 launch` session (not a hypothetical),
running for 666 cycles (~63 seconds wall-clock at 10 Hz, default settings)
before being stopped manually:

- **unit_1** and **unit_4** (both `HEALTHY`) stayed correctly classified as
  `HEALTHY` for the overwhelming majority of the run.
- **unit_3** (`OVERHEATING`) locked onto the correct label almost
  immediately (confidence 0.99–1.00 within the first ~5 cycles of leaving
  warm-up) and held it for the entire run.
- **unit_2** (`BEARING_WEAR`) started with low confidence (0.61–0.63 at
  first detection around cycle 23–26, since wear hadn't developed yet) and
  climbed steadily to a stable 0.97–1.00 by roughly cycle 50 onward — a
  clean example of a gradually-developing fault being detected with
  increasing certainty over time, exactly as a real predictive-maintenance
  system should behave.

### A genuine limitation, observed and not hidden

In the same run, both `HEALTHY` units occasionally flickered to
`SENSOR_DROPOUT` for a few cycles at low confidence (e.g. unit_1 at ~0.79
around cycle 495–498, unit_4 at ~0.53 around cycle 528 — right at the
warning threshold). `unit_2`'s `BEARING_WEAR` confidence also dipped as low
as 0.77–0.78 a few times late in the run despite the unit being genuinely
faulty throughout.

This is not a bug — it is a real consequence of how `SENSOR_DROPOUT` is
defined: the fault fires _intermittently_ (a ~35% chance per cycle after
onset), so a 20-sample rolling window over a genuinely healthy unit can
occasionally contain enough noise to statistically resemble a dropout
glitch. Two things are worth noting about this:

1. The false positives are low-confidence (0.53–0.79), not confident wrong
   answers — the classifier is appropriately uncertain rather than
   confidently mistaken, which is the behavior you want from a monitoring
   system.
2. It reflects a genuine property of intermittent-fault detection in
   general: distinguishing "occasionally noisy but healthy" from
   "occasionally glitching due to a real fault" is an inherently harder
   problem than detecting a fault with a continuous, monotonic signature
   (like `BEARING_WEAR` or `OVERHEATING`). This matches the lower recall
   for `SENSOR_DROPOUT` (0.64) in the held-out test metrics above.

Top features by importance: `vibration_std`, `vibration_mean`, `temp_slope`,
`temp_std` — matches intuition, since bearing wear and overheating are the
two faults with strong, continuous signal drift, which is what a
rolling-window mean/std/slope feature set is best at capturing.

## Reducing terminal flood / GUI monitor option

Two changes address terminal flooding:

1. **In-place terminal redraw**: `dashboard_node`'s status table now
   redraws in place (ANSI cursor-up + line-clear) instead of printing a new
   block every tick. This works cleanly when running `dashboard_node`
   directly in its own terminal:

   ```bash
   ros2 run fleet_health_monitor dashboard_node
   ```

   Under `ros2 launch`, ROS multiplexes and prefixes every process's stdout
   line-by-line (`[dashboard_node-5] ...`), which interferes with in-place
   cursor redraws -- it still works, just less cleanly, since the launch
   log aggregator adds a prefix to every line rather than passing the
   terminal through untouched.

2. **Edge-triggered fault warnings**: `[unit_N] FAULT DETECTED: ...` now
   logs once per state _transition_ (healthy → fault, or fault type
   change) instead of on every single incoming telemetry message. This was
   the main source of terminal flooding previously -- a persistent fault
   was logging a new WARN line at the full 10 Hz telemetry rate.

3. **Standalone GUI monitor** (`gui_monitor.py`): a small Tkinter window,
   one color-coded row per unit, subscribing to `/fleet/status`. Unaffected
   by `ros2 launch`'s stdout multiplexing since it's a separate window, not
   terminal text. Requires `python3-tk` (`sudo apt install python3-tk` if
   not already present). Run alongside the rest of the fleet:
   ```bash
   ros2 run fleet_health_monitor gui_monitor
   ```

## How to run

### 1. (Optional) Generate data + train the model

A pre-trained `models/fault_classifier.pkl` is already included in this
repo, so **you can skip straight to step 2** unless you want to regenerate
the training data or retrain from scratch (e.g. after tweaking
`fault_signatures.py`):

```bash
cd fleet_health_monitor
python3 scripts/generate_training_data.py   # writes data/fleet_training_data.csv
python3 scripts/train_model.py              # writes models/fault_classifier.pkl
```

### 2. Build the ROS 2 package

```bash
# from your ROS 2 workspace src/ folder
colcon build --packages-select fleet_health_monitor
source install/setup.bash
```

### 3. Launch the full demo

```bash
ros2 launch fleet_health_monitor fleet_demo.launch.py
```

To speed up degradation (advances the simulated clock 3 cycles per tick instead of 1):

```bash
ros2 launch fleet_health_monitor fleet_demo.launch.py cycle_step:=3
```

You'll see a live, color-coded terminal table like:

```
unit_1: HEALTHY (conf 0.98, cyc 140) | unit_2: BEARING_WEAR (conf 0.81, cyc 140) | unit_3: HEALTHY (conf 0.95, cyc 140) | unit_4: HEALTHY (conf 0.97, cyc 140)
```

and warnings printed as units cross the fault threshold, e.g.:

```
[unit_2] FAULT DETECTED: BEARING_WEAR (confidence 0.81) at cycle 142
```

The system runs indefinitely (no cycle cap) until stopped with `Ctrl+C`;
shutdown is clean with no tracebacks.

### 4. Inspect the mission log

```bash
cat data/mission_log.csv
```

## Package layout

```
fleet_health_monitor/
  fleet_health_monitor/
    fault_signatures.py         # shared fault signal generation (source of truth)
    feature_engineering.py      # shared rolling-window feature calc
    unit_telemetry_publisher.py # Week 1: one node per fleet unit
    dashboard_node.py           # Week 1 + 3: subscribes to all units, runs live ML
    telemetry_logger.py         # writes /fleet/status to CSV
    gui_monitor.py              # standalone Tkinter live status window
  scripts/
    generate_training_data.py   # offline: builds labeled CSV from fault_signatures
    train_model.py              # offline: trains + evaluates RandomForest, saves .pkl
  launch/
    fleet_demo.launch.py        # single entry point, brings up entire fleet + dashboard
  models/
    fault_classifier.pkl        # pre-trained model
  data/
    fleet_training_data.csv     # generated training data
    mission_log.csv             # generated at runtime
  package.xml / setup.py / setup.cfg
```

## Notes

- Confirm `models/fault_classifier.pkl` exists before launching (dashboard_node
  will fail to start without it — locate it via
  `ament_index_python.packages.get_package_share_directory`, not a relative
  path, since colcon installs it to `share/fleet_health_monitor/models/`).
- The table needs ~20 samples (2s at 10Hz) per unit before predictions start;
  units show "warming up..." until then.
- Ctrl+C shutdown is clean (guarded with `rclpy.ok()` before
  `rclpy.shutdown()` in every node's `main()`), so it exits without a
  traceback appearing on screen.
