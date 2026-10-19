"""The farm on a real map, with Mesa-Geo.

``GeoFarmModel`` is the normal ``FarmModel`` plus a Mesa-Geo ``GeoSpace``. The grid
is laid on the ground in the British National Grid (EPSG:27700, metres), with each
cell ``CELL_METRES`` wide, starting at an anchor point in the countryside north of
Bristol. The anchor is illustrative: change ``ANCHOR`` to put the farm anywhere in
Great Britain. Crop areas, the river and the bases become polygons; robots become
points that follow their grid cells every step.

The robots still plan and move on the grid. Mesa-Geo adds geographic output: the
Leaflet map in ``run_geo.py`` and GeoJSON export (``export_geojson``) for GIS tools
such as QGIS.
"""
import json

import mesa_geo as mg
from shapely.geometry import Point, box

from .drone import ExplorerDrone
from .environment import DRONE_PAD, RIPE
from .model import FarmModel

CRS = "epsg:27700"              # British National Grid (metres)
ANCHOR = (360_500.0, 181_200.0)  # easting, northing of the farm's south-west corner
CELL_METRES = 2.0


class FarmFeature(mg.GeoAgent):
    """A static map feature: a crop area, the river or a base station."""

    def __init__(self, unique_id, model, geometry, kind, patch_id=None):
        super().__init__(unique_id, model, geometry, CRS)
        self.kind = kind
        self.patch_id = patch_id

    def step(self):
        pass


class RobotMarker(mg.GeoAgent):
    """Where a robot is on the map."""

    def __init__(self, unique_id, model, robot):
        super().__init__(unique_id, model, model.cell_point(robot.pos), CRS)
        self.robot = robot

    def step(self):
        pass


class GeoFarmModel(FarmModel):
    def __init__(self, *args, anchor=ANCHOR, cell_metres=CELL_METRES, **kwargs):
        self.anchor = anchor
        self.cell_metres = cell_metres
        super().__init__(*args, **kwargs)
        self.space = mg.GeoSpace(crs=CRS, warn_crs_conversion=False)
        self.features = []
        for patch_id, trees in self.patches.items():
            xs = [t.pos[0] for t in trees]
            ys = [t.pos[1] for t in trees]
            self.features.append(FarmFeature(self.next_id(), self, self.cell_box(min(xs), min(ys), max(xs), max(ys)),
                                             "crop area", patch_id))
        if self.farm.water_cells:
            xs = [x for x, _ in self.farm.water_cells]
            self.features.append(FarmFeature(self.next_id(), self,
                                             self.cell_box(min(xs), 0, max(xs), self.grid.height - 1), "river"))
        for cells, kind in ((self.farm.picker_base, "picker base"), (self.farm.drone_pad, "drone pad")):
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            self.features.append(FarmFeature(self.next_id(), self, self.cell_box(min(xs), min(ys), max(xs), max(ys)),
                                             kind))
        self.markers = [RobotMarker(self.next_id(), self, r) for r in self.pickers + self.drones]
        self.space.add_agents(self.features + self.markers)

    # grid cell -> metres on the ground -------------------------------------- #
    def cell_box(self, x0, y0, x1, y1):
        e, n, c = self.anchor[0], self.anchor[1], self.cell_metres
        return box(e + x0 * c, n + y0 * c, e + (x1 + 1) * c, n + (y1 + 1) * c)

    def cell_point(self, pos):
        e, n, c = self.anchor[0], self.anchor[1], self.cell_metres
        return Point(e + (pos[0] + 0.5) * c, n + (pos[1] + 0.5) * c)

    def area_status(self, patch_id):
        trees = self.patches[patch_id]
        ripe = sum(1 for t in trees if t.state == RIPE)
        return ripe, len(trees)

    def step(self):
        super().step()
        for m in self.markers:
            m.geometry = self.cell_point(m.robot.pos)


def map_portrayal(agent):
    """Leaflet styles for Mesa-Geo's MapModule."""
    if isinstance(agent, RobotMarker):
        robot = agent.robot
        if isinstance(robot, ExplorerDrone):
            return {"radius": 4, "color": "#212121", "fillColor": "#212121", "fillOpacity": 1,
                    "description": f"Drone {robot.unique_id}: {robot.state}, altitude {robot.altitude}"}
        from .portrayal import PICKER_COLOURS

        colour = PICKER_COLOURS.get(robot.state, "#ffffff")
        return {"radius": 6, "color": "#ffffff", "weight": 1, "fillColor": colour, "fillOpacity": 1,
                "description": f"Picker {robot.unique_id}: {robot.state.replace('_', ' ')}, "
                               f"{robot.load}/{robot.capacity} strawberries, battery {robot.battery:.0f}%"}
    if agent.kind == "crop area":
        ripe, total = agent.model.area_status(agent.patch_id)
        share = ripe / total
        colour = "#e53935" if share >= 0.5 else ("#2e7d32" if share == 0 else "#ef9a9a")
        return {"color": "#1b5e20", "weight": 1, "fillColor": colour, "fillOpacity": 0.7,
                "description": f"Crop area {agent.patch_id}: {ripe}/{total} trees ripe"}
    if agent.kind == "river":
        return {"color": "#1e88e5", "weight": 0, "fillColor": "#64b5f6", "fillOpacity": 0.8,
                "description": "River (robots cross slowly)"}
    colour = "#37474f" if agent.kind == DRONE_PAD.replace("_", " ") else "#455a64"
    return {"color": colour, "weight": 1, "fillColor": colour, "fillOpacity": 0.9, "description": agent.kind}


def export_geojson(model, path):
    """Write every feature and robot, in WGS84 (EPSG:4326), as a GeoJSON file."""
    frame = model.space.get_agents_as_GeoDataFrame().to_crs("epsg:4326")
    rows = []
    for agent, geom in zip(model.space.agents, frame.geometry):
        props = {"type": type(agent).__name__}
        if isinstance(agent, RobotMarker):
            props.update(robot=type(agent.robot).__name__, id=agent.robot.unique_id, state=agent.robot.state)
        else:
            props.update(kind=agent.kind)
            if agent.patch_id is not None:
                ripe, total = model.area_status(agent.patch_id)
                props.update(patch=agent.patch_id, ripe=ripe, trees=total)
        rows.append({"type": "Feature", "geometry": geom.__geo_interface__, "properties": props})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": rows}, f)
    return len(rows)


def map_centre(model):
    """(lat, lon) of the farm's centre, for the map view."""
    from pyproj import Transformer

    t = Transformer.from_crs(CRS, "epsg:4326", always_xy=True)
    e = model.anchor[0] + model.grid.width * model.cell_metres / 2
    n = model.anchor[1] + model.grid.height * model.cell_metres / 2
    lon, lat = t.transform(e, n)
    return lat, lon
