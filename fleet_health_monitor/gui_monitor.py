"""
gui_monitor.py

Small Tkinter GUI showing live fleet health, one row per unit, color-coded
the same way as the terminal table. Subscribes to /fleet/status (published
by dashboard_node) rather than the raw telemetry topics, so it stays in
sync with whatever the dashboard is currently classifying.

Unlike the terminal table, this works cleanly no matter how the node is
launched -- ros2 launch multiplexes and prefixes stdout from every process,
which breaks in-place ANSI terminal redraws, but a separate GUI window is
unaffected since it doesn't touch stdout at all.

Requires: python3-tk (Tkinter is part of the Python standard library but on
Debian/Ubuntu it ships as a separate apt package):
    sudo apt install python3-tk

Run (after the rest of the fleet is already running via fleet_demo.launch.py
or on its own):
    ros2 run fleet_health_monitor gui_monitor
"""

import tkinter as tk

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

COLORS = {
    'HEALTHY': '#2ecc71',
    'BEARING_WEAR': '#f1c40f',
    'OVERHEATING': '#e74c3c',
    'SENSOR_DROPOUT': '#9b59b6',
    'WARMUP': '#7f8c8d',
}
TEXT_COLOR_DARK = '#1e1e1e'   # readable on the lighter yellow row
TEXT_COLOR_LIGHT = '#ffffff'
DARK_BG = '#1e1e1e'


class StatusSubscriber(Node):
    """Thin ROS 2 node: just forwards /fleet/status strings to a callback."""

    def __init__(self, on_status):
        super().__init__('gui_monitor')
        self._on_status = on_status
        self.create_subscription(String, '/fleet/status', self._callback, 10)
        self.get_logger().info("GUI monitor subscribed to /fleet/status")

    def _callback(self, msg: String):
        self._on_status(msg.data)


def build_row(root, uid):
    frame = tk.Frame(root, bg=DARK_BG)
    frame.pack(fill='x', padx=12, pady=6)

    name_lbl = tk.Label(
        frame, text=uid, font=("Segoe UI", 13, "bold"),
        fg='white', bg=DARK_BG, width=9, anchor='w')
    name_lbl.pack(side='left')

    status_lbl = tk.Label(
        frame, text="warming up...", font=("Segoe UI", 13),
        fg='white', bg=COLORS['WARMUP'], width=30, anchor='w', padx=10, pady=4)
    status_lbl.pack(side='left', fill='x', expand=True)

    return status_lbl


def main(args=None):
    rclpy.init(args=args)

    root = tk.Tk()
    root.title("Fleet Health Monitor")
    root.configure(bg=DARK_BG)
    root.geometry("460x260")

    title_lbl = tk.Label(
        root, text="Fleet Health Monitor", font=("Segoe UI", 16, "bold"),
        fg='white', bg=DARK_BG, pady=10)
    title_lbl.pack(fill='x')

    row_labels = {}  # unit_id -> tk.Label

    def on_status(data: str):
        # data looks like: "unit_1:HEALTHY:0.98:140 | unit_2:BEARING_WEAR:0.81:139 | ..."
        for part in data.split(" | "):
            fields = part.split(":")
            if len(fields) != 4:
                continue
            uid, label, conf_str, cycle_str = fields
            if uid not in row_labels:
                row_labels[uid] = build_row(root, uid)

            color = COLORS.get(label, COLORS['WARMUP'])
            fg = TEXT_COLOR_DARK if label == 'BEARING_WEAR' else TEXT_COLOR_LIGHT
            try:
                conf = float(conf_str)
            except ValueError:
                conf = 0.0
            try:
                cycle = int(cycle_str)
            except ValueError:
                cycle = -1
            cycle_display = str(cycle) if cycle >= 0 else "--"

            text = "warming up..." if label == 'WARMUP' else f"{label}  (conf {conf:.2f}, cyc {cycle_display})"
            lbl = row_labels[uid]
            # Schedule the actual widget update on Tk's main thread via after(0, ...)
            root.after(0, lambda lbl=lbl, text=text, color=color, fg=fg: (
                lbl.config(text=text, bg=color, fg=fg)))

    node = StatusSubscriber(on_status)

    def spin_once_and_reschedule():
        rclpy.spin_once(node, timeout_sec=0.0)
        root.after(50, spin_once_and_reschedule)

    root.after(50, spin_once_and_reschedule)

    try:
        root.mainloop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
