"""Gazebo side: generate an SDF world of the farm, and keep the robots' poses in sync.

``make_world`` writes ``farm.sdf``: a ground plane, one static box per tree (a single
model with many links), the river, the bases, and one kinematic model per robot
(pickers are red cylinders, drones are dark boxes).

``gz_sync`` subscribes to ``/farm/robot_poses`` (published by ``sim_node``) and moves
every robot model to its simulated position each step, using Gazebo's
``/world/farm/set_pose_vector`` service. Gazebo does not drive the robots: the Mesa
model does the decision making and Gazebo provides the 3D view and physics world.
"""
import argparse
import json
import shutil
import subprocess

from .common import ALTITUDE_METRES, CELL_METRES, cell_to_xy, hex_to_rgb, import_farm


def _box(name, x, y, z, sx, sy, sz, rgb):
    r, g, b = rgb
    return f"""
      <link name="{name}">
        <pose>{x:.2f} {y:.2f} {z:.2f} 0 0 0</pose>
        <visual name="v"><geometry><box><size>{sx:.2f} {sy:.2f} {sz:.2f}</size></box></geometry>
          <material><ambient>{r:.2f} {g:.2f} {b:.2f} 1</ambient><diffuse>{r:.2f} {g:.2f} {b:.2f} 1</diffuse></material>
        </visual>
        <collision name="c"><geometry><box><size>{sx:.2f} {sy:.2f} {sz:.2f}</size></box></geometry></collision>
      </link>"""


def world_sdf(model):
    from farm import portrayal  # noqa: PLC0415

    c = CELL_METRES
    trees = "".join(
        _box(f"tree_{i}", *cell_to_xy(t.pos), 0.9, c * 0.9, c * 0.9, 1.8, hex_to_rgb(portrayal.TREE_GROWING))
        for i, t in enumerate(model.trees)
    )
    river = "".join(
        _box(f"water_{i}", *cell_to_xy(cell), 0.01, c, c, 0.02, hex_to_rgb(portrayal.WATER))
        for i, cell in enumerate(sorted(model.farm.water_cells))
    )
    bases = "".join(
        _box(f"base_{i}", *cell_to_xy(cell), 0.02, c, c, 0.04, hex_to_rgb(portrayal.PICKER_BASE_COLOUR))
        for i, cell in enumerate(model.farm.picker_base + model.farm.drone_pad)
    )
    robots = []
    for r in model.pickers + model.drones:
        is_picker = hasattr(r, "capacity")
        name = f"{'picker' if is_picker else 'drone'}_{r.unique_id}"
        x, y = cell_to_xy(r.pos)
        z = 0.4 if is_picker else r.altitude * ALTITUDE_METRES + 0.5
        geom = ("<cylinder><radius>0.7</radius><length>0.8</length></cylinder>" if is_picker
                else "<box><size>1.0 1.0 0.25</size></box>")
        rgb = "0.85 0.11 0.38" if is_picker else "0.13 0.13 0.13"
        robots.append(f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {z:.2f} 0 0 0</pose>
      <link name="body"><visual name="v"><geometry>{geom}</geometry>
        <material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material></visual></link>
    </model>""")
    w, h = model.grid.width * c, model.grid.height * c
    return f"""<?xml version="1.0"?>
<sdf version="1.8">
  <world name="farm">
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows><pose>0 0 30 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><direction>-0.4 0.2 -0.9</direction>
    </light>
    <model name="ground">
      <static>true</static>
      <pose>{w / 2:.2f} {h / 2:.2f} 0 0 0 0</pose>
      <link name="l">
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>{w + 20:.0f} {h + 20:.0f}</size></plane></geometry></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>{w + 20:.0f} {h + 20:.0f}</size></plane></geometry>
          <material><ambient>0.63 0.53 0.5 1</ambient><diffuse>0.63 0.53 0.5 1</diffuse></material></visual>
      </link>
    </model>
    <model name="crops"><static>true</static>{trees}
    </model>
    <model name="river"><static>true</static>{river}
    </model>
    <model name="bases"><static>true</static>{bases}
    </model>{"".join(robots)}
  </world>
</sdf>
"""


def make_world_main():
    parser = argparse.ArgumentParser(description="Write the farm as a Gazebo SDF world")
    parser.add_argument("--out", default="farm.sdf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pickers", type=int, default=4)
    parser.add_argument("--n-drones", type=int, default=2)
    args, _ = parser.parse_known_args()
    FarmModel = import_farm()
    model = FarmModel(seed=args.seed, n_pickers=args.n_pickers, n_drones=args.n_drones)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(world_sdf(model))
    print(f"wrote {args.out}")


def pose_vector_request(poses):
    """gz.msgs.Pose_V in protobuf text format."""
    items = " ".join(
        f'pose {{ name: "{name}" position {{ x: {x:.3f} y: {y:.3f} z: {z:.3f} }} orientation {{ w: 1 }} }}'
        for name, (x, y, z) in poses.items()
    )
    return items


def sync_main(args=None):
    import rclpy  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from std_msgs.msg import String  # noqa: PLC0415

    tool = shutil.which("gz") or shutil.which("ign")

    class GzSync(Node):
        def __init__(self):
            super().__init__("farm_gz_sync")
            self.declare_parameter("world", "farm")
            self.busy = None
            self.create_subscription(String, "/farm/robot_poses", self.on_poses, 10)
            if tool is None:
                self.get_logger().error("neither 'gz' nor 'ign' is on PATH: cannot move Gazebo models")

        def on_poses(self, msg):
            if tool is None or (self.busy is not None and self.busy.poll() is None):
                return                      # previous update still running: skip this one
            world = self.get_parameter("world").value
            self.busy = subprocess.Popen(
                [tool, "service", "-s", f"/world/{world}/set_pose_vector",
                 "--reqtype", "gz.msgs.Pose_V", "--reptype", "gz.msgs.Boolean",
                 "--timeout", "500", "--req", pose_vector_request(json.loads(msg.data))],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    rclpy.init(args=args)
    node = GzSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
