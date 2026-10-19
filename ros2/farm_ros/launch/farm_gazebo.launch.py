"""Show the farm in Gazebo: generate the world, start Gazebo, run the simulation node
and keep the Gazebo robot models in sync with the Mesa robots.

    ros2 launch farm_ros farm_gazebo.launch.py
"""
import os
import shutil
import tempfile

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    world = os.path.join(tempfile.gettempdir(), "farm.sdf")
    gz = shutil.which("gz") or shutil.which("ign") or "gz"
    sim_cmd = [gz, "sim", "-r", world] if os.path.basename(gz).startswith("gz") else [gz, "gazebo", "-r", world]
    return LaunchDescription([
        ExecuteProcess(cmd=["ros2", "run", "farm_ros", "make_world", "--out", world], output="screen"),
        TimerAction(period=3.0, actions=[ExecuteProcess(cmd=sim_cmd, output="screen")]),
        TimerAction(period=8.0, actions=[
            Node(package="farm_ros", executable="sim_node", name="farm_sim", output="screen",
                 parameters=[{"mode": "Novel", "rate_hz": 4.0}]),
            Node(package="farm_ros", executable="gz_sync", name="farm_gz_sync", output="screen"),
        ]),
    ])
