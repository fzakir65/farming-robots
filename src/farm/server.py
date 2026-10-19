"""Mesa web interface: the farm canvas, live KPI charts and the operator controls.

The ``mode`` drop-down is the switch between the Basic, Extended and Novel modes.
The other drop-downs pick the advanced options (allocation method, picker brain,
path planner, drone strategy, network). Change them, then press *Reset*.

This is the Mesa 2 interface (``run.py``). On Mesa 3 use ``app_solara.py``.
"""
import mesa

from .constants import (
    ALLOCATIONS,
    BASIC,
    BOLD,
    BROADCAST,
    COMM_MODES,
    COMMITMENTS,
    CONTRACT_NET,
    DRONE_STRATEGIES,
    IDEAL,
    INDEPENDENT,
    NETWORKS,
    NOVEL,
    PATH_PLANNERS,
    PICKER_BRAINS,
    ROAM,
    RULES,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MODES,
    PICKER_CAPACITY,
)
from .model import (
    FarmModel,
    blocked_steps,
    energy_per_strawberry,
    harvest_rate,
    mean_detect_latency,
    mean_pick_latency,
    mean_regret,
    overpayment,
    picker_utilisation,
    ripe_on_field,
)
from .portrayal import PICKER_COLOURS, farm_portrayal

CELL_PIXELS = 20


class StatusPanel(mesa.visualization.TextElement):
    """Live KPI summary plus a colour legend."""

    def render(self, model):
        legend = " ".join(
            f'<span style="color:{c}">&#9679;</span>&nbsp;{s.replace("_", " ")}'
            for s, c in PICKER_COLOURS.items()
        )
        extra = ""
        if model.mode != BASIC:
            extra = (
                f" | Messages: <b>{model.bus.sent}</b>"
                f" | Contracts awarded/completed: <b>{model.contracts_awarded}/{model.contracts_completed}</b>"
            )
        if model.mode == NOVEL and model.total_true_cost:
            extra += f" | Regret: <b>{mean_regret(model):.2f}</b> | Overpayment: <b>{overpayment(model):.2f}</b>"
        if getattr(model.bus, "dropped", 0):
            extra += f" | Messages lost: <b>{model.bus.dropped}</b>"
        if hasattr(model.bus, "mean_latency_ms"):
            extra += f" | XMPP latency: <b>{model.bus.mean_latency_ms:.1f} ms</b>"
        if model.llm is not None:
            extra += f" | LLM calls: <b>{model.llm.calls}</b> ({model.llm.seconds:.0f} s, {model.llm_fallbacks} fallbacks)"
        setup = (
            f"<small>Brain: {model.picker_brain} | Paths: {model.path_planner} | Drones: {model.drone_strategy}"
            f" | Network: {model.network}</small><br>"
        )
        return (
            f"<b>Mode:</b> {model.mode}"
            + (f" ({model.comm_mode})" if model.mode == "Extended" else "")
            + (f" ({model.allocation})" if model.mode == NOVEL else "")
            + f" | <b>Step:</b> {model.schedule.steps}<br>"
            f"Delivered: <b>{model.delivered}</b> | Picked: <b>{model.picked}</b>"
            f" | Wasted: <b>{model.wasted}</b> | Ripe on field: <b>{ripe_on_field(model)}</b>"
            f" | Harvest rate: <b>{harvest_rate(model):.0%}</b><br>"
            f"Energy per strawberry: <b>{energy_per_strawberry(model):.2f}</b>"
            f" | Mean pick latency: <b>{mean_pick_latency(model):.1f}</b> steps"
            f" | Picker utilisation: <b>{picker_utilisation(model):.0%}</b>"
            f" | Detection latency: <b>{mean_detect_latency(model):.1f}</b>"
            f" | Blocked steps: <b>{blocked_steps(model)}</b>{extra}<br>{setup}"
            f"<small>Pickers (number = strawberries carried): {legend}<br>"
            f"Drones: black squares (z = altitude). Trees: "
            f'<span style="color:#e53935">ripe</span>, <span style="color:#2e7d32">growing</span>, '
            f'<span style="color:#9ccc65">empty</span>. B = picker base, H = drone pad.</small>'
        )


canvas = mesa.visualization.CanvasGrid(
    farm_portrayal,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH * CELL_PIXELS,
    DEFAULT_HEIGHT * CELL_PIXELS,
)

harvest_chart = mesa.visualization.ChartModule(
    [
        {"Label": "Delivered", "Color": "#2e7d32"},
        {"Label": "Wasted", "Color": "#6d4c41"},
        {"Label": "Ripe on field", "Color": "#e53935"},
    ],
    data_collector_name="datacollector",
)

battery_chart = mesa.visualization.ChartModule(
    [
        {"Label": "Mean picker battery", "Color": "#1e88e5"},
        {"Label": "Mean drone battery", "Color": "#212121"},
    ],
    data_collector_name="datacollector",
)

model_params = {
    "mode": mesa.visualization.Choice("Operation mode", value=BASIC, choices=MODES),
    "comm_mode": mesa.visualization.Choice(
        "Extended mode: drone messaging", value=BROADCAST, choices=COMM_MODES
    ),
    "n_pickers": mesa.visualization.Slider("Picker robots", 4, 1, 10, 1),
    "n_drones": mesa.visualization.Slider("Explorer drones", 2, 1, 6, 1),
    "picker_capacity": mesa.visualization.Slider("Picker storage capacity", PICKER_CAPACITY, 2, 20, 1),
    "initial_ripe_patches": mesa.visualization.Slider("Crop areas ripe at start", 8, 1, 30, 1),
    "random_layout": mesa.visualization.Checkbox("Random layout", False),
    "allocation": mesa.visualization.Choice("Novel mode: allocation", value=CONTRACT_NET, choices=ALLOCATIONS),
    "strategic_bidders": mesa.visualization.Slider("Share of self-interested bidders", 0.0, 0.0, 1.0, 0.25),
    "picker_brain": mesa.visualization.Choice("Picker decision making", value=RULES, choices=PICKER_BRAINS),
    "bdi_commitment": mesa.visualization.Choice("BDI commitment strategy", value=BOLD, choices=COMMITMENTS),
    "path_planner": mesa.visualization.Choice("Path planning", value=INDEPENDENT, choices=PATH_PLANNERS),
    "drone_strategy": mesa.visualization.Choice("Drone search strategy", value=ROAM, choices=DRONE_STRATEGIES),
    "network": mesa.visualization.Choice("Network", value=IDEAL, choices=NETWORKS),
    "drop_prob": mesa.visualization.Slider("Lossy radio: drop probability", 0.1, 0.0, 0.5, 0.05),
}

server = mesa.visualization.ModularServer(
    FarmModel,
    [canvas, StatusPanel(), harvest_chart, battery_chart],
    "Robot-Assisted Farming",
    model_params,
)
