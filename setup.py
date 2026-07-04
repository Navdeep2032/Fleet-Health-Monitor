import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'fleet_health_monitor'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'models'),
            glob(os.path.join('models', '*.pkl'))),
    ],
    install_requires=['setuptools', 'numpy', 'scikit-learn', 'joblib'],
    zip_safe=True,
    maintainer='Navdeep',
    maintainer_email='you@example.com',
    description='Fleet-style multi-node health dashboard with ML fault classification',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'unit_telemetry_publisher = fleet_health_monitor.unit_telemetry_publisher:main',
            'dashboard_node = fleet_health_monitor.dashboard_node:main',
            'telemetry_logger = fleet_health_monitor.telemetry_logger:main',
            'gui_monitor = fleet_health_monitor.gui_monitor:main',
        ],
    },
)
