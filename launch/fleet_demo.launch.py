"""
fleet_demo.launch.py

Single launch file bringing up the full fleet health demo:
  - 4x unit_telemetry_publisher, each a different fault_type
  - 1x dashboard_node (loads the trained classifier, prints live fleet table)
  - 1x telemetry_logger (writes /fleet/status to a CSV mission log)

Run:
    ros2 launch fleet_health_monitor fleet_demo.launch.py

Speed up the demo (advance the simulated clock faster per publish tick):
    ros2 launch fleet_health_monitor fleet_demo.launch.py cycle_step:=3
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cycle_step_arg = DeclareLaunchArgument(
        'cycle_step', default_value='1',
        description='How many simulated cycles advance per publish tick '
                    '(raise to speed up degradation for a short demo).')

    cycle_step = LaunchConfiguration('cycle_step')

    # unit_id -> fault_type assignment for the demo fleet.
    # Two units stay healthy, one develops bearing wear, one overheats --
    # gives a clear "some fail, some don't" story for the demo video.
    unit_configs = [
        ('unit_1', 'HEALTHY'),
        ('unit_2', 'BEARING_WEAR'),
        ('unit_3', 'OVERHEATING'),
        ('unit_4', 'HEALTHY'),
    ]

    unit_ids = [uid for uid, _ in unit_configs]

    nodes = [cycle_step_arg]

    for unit_id, fault_type in unit_configs:
        nodes.append(Node(
            package='fleet_health_monitor',
            executable='unit_telemetry_publisher',
            name=f'{unit_id}_publisher',
            parameters=[{
                'unit_id': unit_id,
                'fault_type': fault_type,
                'publish_hz': 10.0,
                'cycle_step': ParameterValue(cycle_step, value_type=int),
            }],
            output='screen',
        ))

    nodes.append(Node(
        package='fleet_health_monitor',
        executable='dashboard_node',
        name='dashboard_node',
        parameters=[{
            'unit_ids': unit_ids,
            'print_hz': 2.0,
        }],
        output='screen',
    ))

    nodes.append(Node(
        package='fleet_health_monitor',
        executable='telemetry_logger',
        name='telemetry_logger',
        output='screen',
    ))

    return LaunchDescription(nodes)
