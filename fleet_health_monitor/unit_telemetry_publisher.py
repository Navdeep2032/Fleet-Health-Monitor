"""
unit_telemetry_publisher.py

Week 1 (ROS 2 plumbing) piece.

Simulates ONE fleet unit (a motor / subsystem). Launched multiple times with
different `unit_id` and `fault_type` params to build a namespaced fleet of
independent nodes -- this is the multi-node plumbing half of the project.

Publishes on: /<unit_id>/telemetry  (std_msgs/Float32MultiArray)
    data = [current, vibration, temp, cycle]

Params:
    unit_id     (string)  default "unit_1"
    fault_type  (string)  one of HEALTHY / BEARING_WEAR / OVERHEATING /
                          SENSOR_DROPOUT   default "HEALTHY"
    publish_hz  (double)  default 10.0
    cycle_step  (int)     how much the internal "cycle" clock advances per
                          tick; raise this to speed up degradation for a
                          short demo video. default 1
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from fleet_health_monitor.fault_signatures import generate_sample, FAULT_TYPES


class UnitTelemetryPublisher(Node):

    def __init__(self):
        super().__init__('unit_telemetry_publisher')

        self.declare_parameter('unit_id', 'unit_1')
        self.declare_parameter('fault_type', 'HEALTHY')
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('cycle_step', 1)

        self.unit_id = self.get_parameter('unit_id').get_parameter_value().string_value
        self.fault_type = self.get_parameter('fault_type').get_parameter_value().string_value
        publish_hz = self.get_parameter('publish_hz').get_parameter_value().double_value
        self.cycle_step = self.get_parameter('cycle_step').get_parameter_value().integer_value

        if self.fault_type not in FAULT_TYPES:
            self.get_logger().warn(
                f"Unknown fault_type '{self.fault_type}', defaulting to HEALTHY. "
                f"Valid options: {FAULT_TYPES}")
            self.fault_type = 'HEALTHY'

        self.cycle = 0
        self.rng = np.random.default_rng(seed=hash(self.unit_id) % (2**32))

        topic = f'/{self.unit_id}/telemetry'
        self.publisher_ = self.create_publisher(Float32MultiArray, topic, 10)

        period = 1.0 / publish_hz
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f"[{self.unit_id}] publisher started | fault_type={self.fault_type} "
            f"| topic={topic} | {publish_hz} Hz")

    def timer_callback(self):
        current, vibration, temp = generate_sample(self.cycle, self.fault_type, self.rng)

        msg = Float32MultiArray()
        msg.data = [current, vibration, temp, float(self.cycle)]
        self.publisher_.publish(msg)

        self.cycle += self.cycle_step


def main(args=None):
    rclpy.init(args=args)
    node = UnitTelemetryPublisher()
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
