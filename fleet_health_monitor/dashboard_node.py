"""
dashboard_node.py

Fuses Week 1 (multi-node plumbing -- subscribes to N independent unit
topics) with Week 3 (ML -- runs a trained multi-class classifier per unit
on a rolling feature window) to produce a live fleet health report.

Subscribes to: /<unit_id>/telemetry for each unit_id in the `unit_ids` param
Publishes:     /fleet/status  (std_msgs/String, human-readable table)

Params:
    unit_ids     (string array) default ["unit_1","unit_2","unit_3","unit_4"]
    model_path   (string) path to the trained .pkl classifier
    print_hz     (double) how often to refresh the terminal table
"""

import collections
import os
import sys

import joblib
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from fleet_health_monitor.feature_engineering import compute_features, WINDOW


def _default_model_path():
    """
    Locate the installed model via the package's share directory (the
    correct ROS 2 way), falling back to a source-tree-relative path if the
    package isn't found via ament_index (e.g. running the .py file directly
    without colcon build/install).
    """
    try:
        share_dir = get_package_share_directory('fleet_health_monitor')
        return os.path.join(share_dir, 'models', 'fault_classifier.pkl')
    except Exception:
        return os.path.join(
            os.path.dirname(__file__), '..', 'models', 'fault_classifier.pkl')

# ANSI colors for terminal readability
COLOR = {
    'HEALTHY': '\033[92m',        # green
    'BEARING_WEAR': '\033[93m',   # yellow
    'OVERHEATING': '\033[91m',    # red
    'SENSOR_DROPOUT': '\033[95m', # magenta
    'WARMUP': '\033[90m',         # grey
}
RESET = '\033[0m'


class UnitBuffer:
    """Rolling window of raw readings for one unit."""

    def __init__(self, window=WINDOW):
        self.current = collections.deque(maxlen=window)
        self.vibration = collections.deque(maxlen=window)
        self.temp = collections.deque(maxlen=window)
        self.last_cycle = None

    def add(self, current, vibration, temp, cycle):
        self.current.append(current)
        self.vibration.append(vibration)
        self.temp.append(temp)
        self.last_cycle = cycle

    def ready(self, window=WINDOW):
        return len(self.current) == window

    def features(self):
        return compute_features(self.current, self.vibration, self.temp)


class DashboardNode(Node):

    def __init__(self):
        super().__init__('dashboard_node')

        self.declare_parameter(
            'unit_ids', ['unit_1', 'unit_2', 'unit_3', 'unit_4'])
        self.declare_parameter('model_path', _default_model_path())
        self.declare_parameter('print_hz', 2.0)

        self.unit_ids = list(
            self.get_parameter('unit_ids').get_parameter_value().string_array_value)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        print_hz = self.get_parameter('print_hz').get_parameter_value().double_value

        self.get_logger().info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        self.buffers = {uid: UnitBuffer() for uid in self.unit_ids}
        self.latest_prediction = {uid: ('WARMUP', 0.0) for uid in self.unit_ids}
        self._last_logged_label = {uid: None for uid in self.unit_ids}
        self._lines_printed = 0
        self._first_draw = True

        self.subs = []
        for uid in self.unit_ids:
            topic = f'/{uid}/telemetry'
            sub = self.create_subscription(
                Float32MultiArray, topic,
                self._make_callback(uid), 10)
            self.subs.append(sub)
            self.get_logger().info(f"Subscribed to {topic}")

        self.status_pub = self.create_publisher(String, '/fleet/status', 10)
        self.timer = self.create_timer(1.0 / print_hz, self.print_status)

        self.get_logger().info(
            f"Dashboard node started, monitoring {len(self.unit_ids)} units.")

    def _make_callback(self, unit_id):
        def callback(msg: Float32MultiArray):
            current, vibration, temp, cycle = msg.data
            buf = self.buffers[unit_id]
            buf.add(current, vibration, temp, cycle)

            if buf.ready():
                feats = [buf.features()]
                pred_class = self.model.predict(feats)[0]
                proba = self.model.predict_proba(feats)[0]
                confidence = float(max(proba))
                self.latest_prediction[unit_id] = (pred_class, confidence)

                # Edge-triggered: only log when this unit's state actually
                # CHANGES, not on every single incoming message. Without
                # this, a persistent fault logs a new WARN line at the
                # full telemetry rate (10 Hz per unit) and floods the
                # terminal within seconds.
                is_fault = pred_class != 'HEALTHY' and confidence > 0.6
                changed = pred_class != self._last_logged_label[unit_id]
                if is_fault and changed:
                    self.get_logger().warn(
                        f"[{unit_id}] FAULT DETECTED: {pred_class} "
                        f"(confidence {confidence:.2f}) at cycle {int(cycle)}")
                    self._last_logged_label[unit_id] = pred_class
                elif not is_fault and self._last_logged_label[unit_id] is not None:
                    self.get_logger().info(
                        f"[{unit_id}] recovered to HEALTHY at cycle {int(cycle)}")
                    self._last_logged_label[unit_id] = None

        return callback

    def print_status(self):
        parts = []
        lines = []
        for uid in self.unit_ids:
            label, conf = self.latest_prediction[uid]
            color = COLOR.get(label, '')
            cycle = self.buffers[uid].last_cycle
            cycle_str = f"{int(cycle)}" if cycle is not None else "--"
            if label == 'WARMUP':
                display = f"{color}{uid}: warming up...{RESET}"
            else:
                display = f"{color}{uid}: {label} (conf {conf:.2f}, cyc {cycle_str}){RESET}"
            lines.append(display)
            cycle_num = int(cycle) if cycle is not None else -1
            parts.append(f"{uid}:{label}:{conf:.2f}:{cycle_num}")

        # In-place redraw: move the cursor back up to the start of the
        # block we printed last time, clear each line, then print fresh
        # values on top -- instead of printing a brand-new block below the
        # previous one every tick (which is what was flooding the terminal).
        if not self._first_draw:
            sys.stdout.write(f"\033[{self._lines_printed}F")
        for line in lines:
            sys.stdout.write("\033[2K" + line + "\n")
        sys.stdout.flush()

        self._lines_printed = len(lines)
        self._first_draw = False

        status_msg = String()
        status_msg.data = " | ".join(parts)
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
