"""ROS 2 node that runs the farm simulation and publishes it.

Topics (all in the ``map`` frame, metres; one grid cell = 2 m):
    /farm/markers        visualization_msgs/MarkerArray  trees (by crop state), river, bases, robots
    /farm/pickers        geometry_msgs/PoseArray          picker positions
    /farm/drones         geometry_msgs/PoseArray          drone positions (z = altitude)
    /farm/robot_poses    std_msgs/String                  JSON {model name: [x, y, z]} for Gazebo
    /farm/kpis           std_msgs/String                  JSON with the live KPIs
TF: map -> picker_<id>, map -> drone_<id>
Service: ~/reset (std_srvs/Trigger) rebuilds the model with the current parameters.

Parameters: mode, comm_mode, allocation, picker_brain, path_planner, drone_strategy,
n_pickers, n_drones, seed, rate_hz, max_steps.
"""
import json

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .common import ALTITUDE_METRES, CELL_METRES, cell_to_xy, hex_to_rgb, import_farm

STRING_PARAMS = {
    "mode": "Novel",
    "comm_mode": "Broadcast",
    "allocation": "Contract Net (first-price)",
    "picker_brain": "Rule-based",
    "path_planner": "Independent (Dijkstra)",
    "drone_strategy": "Random roaming",
}
INT_PARAMS = {"n_pickers": 4, "n_drones": 2, "seed": 42, "max_steps": 100000}


def colour(hex_colour, alpha=1.0):
    r, g, b = hex_to_rgb(hex_colour)
    return ColorRGBA(r=r, g=g, b=b, a=alpha)


def robot_name(robot):
    return f"{'picker' if hasattr(robot, 'capacity') else 'drone'}_{robot.unique_id}"


class FarmSimNode(Node):
    def __init__(self):
        super().__init__("farm_sim")
        for name, default in {**STRING_PARAMS, **INT_PARAMS}.items():
            self.declare_parameter(name, default)
        self.declare_parameter("rate_hz", 5.0)
        self.FarmModel = import_farm()
        from farm import portrayal  # noqa: PLC0415  (available once the farm package is importable)

        self.portrayal = portrayal
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.markers = self.create_publisher(MarkerArray, "/farm/markers", latched)
        self.pickers_pub = self.create_publisher(PoseArray, "/farm/pickers", 10)
        self.drones_pub = self.create_publisher(PoseArray, "/farm/drones", 10)
        self.poses_pub = self.create_publisher(String, "/farm/robot_poses", 10)
        self.kpis_pub = self.create_publisher(String, "/farm/kpis", 10)
        self.tf = TransformBroadcaster(self)
        self.create_service(Trigger, "~/reset", self.on_reset)
        self.build()
        self.timer = self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.tick)

    # ------------------------------------------------------------------ #
    def build(self):
        kwargs = {k: self.get_parameter(k).value for k in {**STRING_PARAMS, **INT_PARAMS}}
        self.model = self.FarmModel(**kwargs)
        self.get_logger().info(
            f"farm model: {kwargs['mode']}, {kwargs['n_pickers']} pickers, {kwargs['n_drones']} drones"
        )
        self.publish()

    def on_reset(self, request, response):
        self.build()
        response.success = True
        response.message = "farm model rebuilt"
        return response

    def tick(self):
        if self.model.running:
            self.model.step()
        self.publish()

    # ------------------------------------------------------------------ #
    def robot_pose(self, robot):
        x, y = cell_to_xy(robot.pos)
        z = robot.altitude * ALTITUDE_METRES + 0.5 if hasattr(robot, "altitude") else 0.4
        return x, y, z

    def publish(self):
        model = self.model
        stamp = self.get_clock().now().to_msg()
        self.markers.publish(self.marker_array(stamp))

        poses = {}
        for pub, robots in ((self.pickers_pub, model.pickers), (self.drones_pub, model.drones)):
            msg = PoseArray()
            msg.header.frame_id = "map"
            msg.header.stamp = stamp
            for r in robots:
                x, y, z = self.robot_pose(r)
                p = Pose()
                p.position.x, p.position.y, p.position.z = x, y, z
                p.orientation.w = 1.0
                msg.poses.append(p)
                poses[robot_name(r)] = [x, y, z]
                t = TransformStamped()
                t.header.stamp = stamp
                t.header.frame_id = "map"
                t.child_frame_id = robot_name(r)
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
                t.transform.rotation.w = 1.0
                self.tf.sendTransform(t)
            pub.publish(msg)
        self.poses_pub.publish(String(data=json.dumps(poses)))

        from farm.model import energy_per_strawberry, harvest_rate, mean_pick_latency  # noqa: PLC0415

        kpis = {
            "step": model.schedule.steps,
            "delivered": model.delivered,
            "wasted": model.wasted,
            "harvest_rate": round(harvest_rate(model), 3),
            "energy_per_strawberry": round(energy_per_strawberry(model), 2),
            "mean_pick_latency": round(mean_pick_latency(model), 1),
            "messages": model.bus.sent,
        }
        self.kpis_pub.publish(String(data=json.dumps(kpis)))

    def marker_array(self, stamp):
        model, port = self.model, self.portrayal
        arr = MarkerArray()

        def cube_list(ns, mid, cells, colours, height, z=0.0, scale=0.9):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = ns, mid, Marker.CUBE_LIST, Marker.ADD
            m.scale.x = m.scale.y = CELL_METRES * scale
            m.scale.z = height
            m.pose.orientation.w = 1.0
            for cell, c in zip(cells, colours):
                x, y = cell_to_xy(cell)
                m.points.append(Point(x=x, y=y, z=z + height / 2))
                m.colors.append(c)
            return m

        trees = model.trees
        tree_colours = [colour({"ripe": port.TREE_RIPE, "growing": port.TREE_GROWING}.get(t.state, port.TREE_EMPTY))
                        for t in trees]
        arr.markers.append(cube_list("trees", 0, [t.pos for t in trees], tree_colours, 1.2))
        water = sorted(model.farm.water_cells)
        arr.markers.append(cube_list("river", 1, water, [colour(port.WATER, 0.8)] * len(water), 0.05, scale=1.0))
        bases = model.farm.picker_base + model.farm.drone_pad
        arr.markers.append(cube_list("bases", 2, bases, [colour(port.PICKER_BASE_COLOUR)] * len(bases), 0.1,
                                     scale=1.0))

        pickers = Marker()
        pickers.header.frame_id = "map"
        pickers.header.stamp = stamp
        pickers.ns, pickers.id, pickers.type = "pickers", 3, Marker.SPHERE_LIST
        pickers.scale.x = pickers.scale.y = pickers.scale.z = CELL_METRES * 0.8
        pickers.pose.orientation.w = 1.0
        for p in model.pickers:
            x, y, z = self.robot_pose(p)
            pickers.points.append(Point(x=x, y=y, z=z))
            pickers.colors.append(colour(port.PICKER_COLOURS.get(p.state, "#ffffff")))
        arr.markers.append(pickers)

        drones = Marker()
        drones.header.frame_id = "map"
        drones.header.stamp = stamp
        drones.ns, drones.id, drones.type = "drones", 4, Marker.CUBE_LIST
        drones.scale.x = drones.scale.y = CELL_METRES * 0.6
        drones.scale.z = 0.3
        drones.pose.orientation.w = 1.0
        for d in model.drones:
            x, y, z = self.robot_pose(d)
            drones.points.append(Point(x=x, y=y, z=z))
            drones.colors.append(colour("#212121"))
        arr.markers.append(drones)

        text = Marker()
        text.header.frame_id = "map"
        text.header.stamp = stamp
        text.ns, text.id, text.type = "kpis", 5, Marker.TEXT_VIEW_FACING
        text.pose.position.x = model.grid.width * CELL_METRES / 2
        text.pose.position.y = model.grid.height * CELL_METRES + 4
        text.pose.position.z = 4.0
        text.pose.orientation.w = 1.0
        text.scale.z = 2.0
        text.color = colour("#ffffff")
        text.text = (f"{model.mode} | step {model.schedule.steps} | delivered {model.delivered} "
                     f"| wasted {model.wasted}")
        arr.markers.append(text)
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = FarmSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.model.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
