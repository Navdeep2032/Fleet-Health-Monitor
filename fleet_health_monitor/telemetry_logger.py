"""
telemetry_logger.py

Optional node: subscribes to /fleet/status (published by dashboard_node)
and appends timestamped rows to a mission log CSV. Useful for the README /
grading evidence -- gives you a plain-text record of the whole demo run
without needing to scrape the terminal.

Params:
    log_path (string) default data/mission_log.csv
"""

import csv
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TelemetryLogger(Node):

    def __init__(self):
        super().__init__('telemetry_logger')

        self.declare_parameter(
            'log_path',
            os.path.join(os.path.dirname(__file__), '..', 'data', 'mission_log.csv'))
        self.log_path = self.get_parameter('log_path').get_parameter_value().string_value

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._file_is_new = not os.path.exists(self.log_path)
        self.csv_file = open(self.log_path, 'a', newline='')
        self.writer = csv.writer(self.csv_file)
        if self._file_is_new:
            self.writer.writerow(['wall_time', 'fleet_status'])

        self.sub = self.create_subscription(
            String, '/fleet/status', self.callback, 10)

        self.get_logger().info(f"Logging /fleet/status to {self.log_path}")

    def callback(self, msg: String):
        self.writer.writerow([time.time(), msg.data])
        self.csv_file.flush()

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryLogger()
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
