"""Run the farm simulation as a ROS 2 node and show it in RViz2.

    ros2 launch farm_ros farm_rviz.launch.py mode:=Novel n_pickers:=6
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {"mode": "Novel", "allocation": "Contract Net (first-price)", "n_pickers": "4", "n_drones": "2",
        "seed": "42", "rate_hz": "5.0", "rviz": "true"}


def generate_launch_description():
    share = get_package_share_directory("farm_ros")
    params = {k: LaunchConfiguration(k) for k in ("mode", "allocation", "seed", "n_pickers", "n_drones", "rate_hz")}
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
        + [
            Node(package="farm_ros", executable="sim_node", name="farm_sim", output="screen",
                 parameters=[{
                     "mode": params["mode"], "allocation": params["allocation"],
                     "seed": params["seed"], "n_pickers": params["n_pickers"],
                     "n_drones": params["n_drones"], "rate_hz": params["rate_hz"],
                 }]),
            Node(package="rviz2", executable="rviz2", name="rviz2", output="log",
                 arguments=["-d", os.path.join(share, "rviz", "farm.rviz")],
                 condition=IfCondition(LaunchConfiguration("rviz"))),
        ]
    )
