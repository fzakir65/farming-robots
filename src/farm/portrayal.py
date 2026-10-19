"""How each agent is drawn on the Mesa canvas."""
from .drone import ExplorerDrone
from .environment import DRONE_PAD, GROWING, RIPE, Base, Ground, Tree, Water
from .picker import (
    AT_BASE,
    CHARGING,
    EXPLORING,
    MOVING_TO_CROP,
    OUT_OF_BATTERY,
    PICKING,
    RETURNING,
    PickerRobot,
)

SOIL = "#a1887f"
WATER = "#64b5f6"
PICKER_BASE_COLOUR = "#455a64"
DRONE_PAD_COLOUR = "#37474f"
TREE_GROWING = "#2e7d32"
TREE_RIPE = "#e53935"
TREE_EMPTY = "#c5e1a5"

PICKER_COLOURS = {
    AT_BASE: "#9e9e9e",
    CHARGING: "#fdd835",
    EXPLORING: "#ff9800",
    MOVING_TO_CROP: "#1e88e5",
    PICKING: "#d81b60",
    RETURNING: "#6a1b9a",
    OUT_OF_BATTERY: "#000000",
}


def cell(colour, layer=0, size=1.0):
    return {"Shape": "rect", "w": size, "h": size, "Filled": "true", "Layer": layer, "Color": colour}


def farm_portrayal(agent):
    if isinstance(agent, Ground):
        return cell(SOIL)
    if isinstance(agent, Water):
        return cell(WATER)
    if isinstance(agent, Base):
        p = cell(DRONE_PAD_COLOUR if agent.kind == DRONE_PAD else PICKER_BASE_COLOUR)
        p["text"] = "H" if agent.kind == DRONE_PAD else "B"
        p["text_color"] = "white"
        return p
    if isinstance(agent, Tree):
        colour = {RIPE: TREE_RIPE, GROWING: TREE_GROWING}.get(agent.state, TREE_EMPTY)
        return cell(colour, size=0.92)
    if isinstance(agent, PickerRobot):
        return {
            "Shape": "circle",
            "r": 0.85,
            "Filled": "true",
            "Layer": 1,
            "Color": PICKER_COLOURS.get(agent.state, "white"),
            "text": str(agent.load),
            "text_color": "white",
        }
    if isinstance(agent, ExplorerDrone):
        return {
            "Shape": "rect",
            "w": 0.55,
            "h": 0.55,
            "Filled": "true",
            "Layer": 2,
            "Color": "#212121" if agent.is_active else "#757575",
            "text": f"z{agent.altitude}",
            "text_color": "white",
        }
    return None
