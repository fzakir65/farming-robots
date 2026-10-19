from glob import glob

from setuptools import setup

package_name = "farm_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AA-MAS student",
    maintainer_email="student@example.com",
    description="ROS 2 bridge for the robot-assisted farming MAS",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_node = farm_ros.sim_node:main",
            "make_world = farm_ros.gz_world:make_world_main",
            "gz_sync = farm_ros.gz_world:sync_main",
        ],
    },
)
