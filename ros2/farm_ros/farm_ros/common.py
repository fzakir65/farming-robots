"""Shared helpers: find the Mesa model, build it from ROS parameters, map cells to metres."""
import os
import sys

CELL_METRES = 2.0       # one grid cell on the ground (same as the Mesa-Geo view)
ALTITUDE_METRES = 1.5   # one drone altitude level


def farm_src():
    """Path of farming-robots/src (parameter, FARM_SRC, or found relative to this package)."""
    candidates = [os.environ.get("FARM_SRC", "")]
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(8):
        candidates.append(os.path.join(here, "src"))
        here = os.path.dirname(here)
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "farm", "model.py")):
            return c
    raise RuntimeError("cannot find farming-robots/src: set the FARM_SRC environment variable")


def import_farm(src=None):
    src = src or farm_src()
    if src not in sys.path:
        sys.path.insert(0, src)
    from farm.model import FarmModel  # noqa: PLC0415

    return FarmModel


def cell_to_xy(pos):
    """Centre of a grid cell in metres (x east, y north) in the ``map`` frame."""
    return (pos[0] + 0.5) * CELL_METRES, (pos[1] + 0.5) * CELL_METRES


def hex_to_rgb(colour):
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
