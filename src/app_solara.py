"""Mesa 3 web interface (Solara). The same model as ``run.py``, on the new Mesa API.

Run it from ``src/`` with the Mesa 3 environment:
    ..\\.venv-mesa3\\Scripts\\solara run app_solara.py      (opens http://localhost:8765)

The Mesa 2 interface (``run.py``, ModularServer) is what the module uses. This file
shows the same farm running on Mesa 3.x with ``SolaraViz``, without changing the model.

The farm is drawn by our own matplotlib component (``FarmSpace``): in Mesa 3.5.1,
``make_space_component``'s matplotlib backend ignores per-agent marker sizes, so
robots were drawn as tiny dots.
"""
import solara
from matplotlib.figure import Figure
from mesa.visualization import SolaraViz, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.utils import update_counter

from farm.constants import (ALLOCATIONS, BASIC, COMM_MODES, CONTRACT_NET, DRONE_STRATEGIES, INDEPENDENT,
                            MODES, PATH_PLANNERS, PICKER_BRAINS, ROAM, RULES)
from farm.drone import ExplorerDrone
from farm.environment import DRONE_PAD, GROWING, RIPE, Base, Ground, Tree, Water
from farm.model import FarmModel, harvest_rate, mean_detect_latency, mean_pick_latency
from farm.picker import PickerRobot
from farm.portrayal import (DRONE_PAD_COLOUR, PICKER_BASE_COLOUR, PICKER_COLOURS, SOIL, TREE_EMPTY,
                            TREE_GROWING, TREE_RIPE, WATER)


def portrayal(agent):
    if isinstance(agent, PickerRobot):
        return AgentPortrayalStyle(color=PICKER_COLOURS.get(agent.state, "white"), marker="o", size=170,
                                   zorder=3, edgecolors="white", linewidths=1)
    if isinstance(agent, ExplorerDrone):
        return AgentPortrayalStyle(color="#212121", marker="s", size=110, zorder=4, edgecolors="white", linewidths=1)
    if isinstance(agent, Tree):
        colour = {RIPE: TREE_RIPE, GROWING: TREE_GROWING}.get(agent.state, TREE_EMPTY)
        return AgentPortrayalStyle(color=colour, marker="s", size=150, zorder=1)
    if isinstance(agent, Water):
        return AgentPortrayalStyle(color=WATER, marker="s", size=150, zorder=0)
    if isinstance(agent, Base):
        colour = DRONE_PAD_COLOUR if agent.kind == DRONE_PAD else PICKER_BASE_COLOUR
        return AgentPortrayalStyle(color=colour, marker="s", size=150, zorder=0)
    if isinstance(agent, Ground):
        return AgentPortrayalStyle(color=SOIL, marker="s", size=150, zorder=0, alpha=0.5)
    return AgentPortrayalStyle(size=0)


@solara.component
def FarmSpace(model):
    """The farm grid: terrain and trees underneath, pickers and drones on top."""
    update_counter.get()
    fig = Figure(figsize=(6.4, 6))
    ax = fig.subplots()
    layers = {}
    for contents, (x, y) in model.grid.coord_iter():
        for agent in contents:
            style = portrayal(agent)
            if style.size:
                key = (style.zorder, style.marker)
                layers.setdefault(key, []).append((x, y, style))
    for (zorder, marker), items in sorted(layers.items()):
        ax.scatter([i[0] for i in items], [i[1] for i in items], marker=marker, zorder=zorder,
                   s=[i[2].size for i in items], c=[i[2].color for i in items],
                   alpha=items[0][2].alpha, edgecolors=items[0][2].edgecolors or "face",
                   linewidths=items[0][2].linewidths)
    ax.set_xlim(-0.5, model.grid.width - 0.5)
    ax.set_ylim(-0.5, model.grid.height - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def Status(model):
    update_counter.get()
    solara.Markdown(
        f"**{model.mode}** · step {model.schedule.steps}  \n"
        f"Delivered **{model.delivered}** · wasted **{model.wasted}** · "
        f"harvest rate **{harvest_rate(model):.0%}**  \n"
        f"Pick latency **{mean_pick_latency(model):.1f}** · detection latency "
        f"**{mean_detect_latency(model):.1f}** · messages **{model.bus.sent}**"
    )


def choice(label, values, value):
    return {"type": "Select", "value": value, "values": values, "label": label}


model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "Random seed"},
    "mode": choice("Operation mode", MODES, BASIC),
    "comm_mode": choice("Extended mode: drone messaging", COMM_MODES, COMM_MODES[0]),
    "allocation": choice("Novel mode: allocation", ALLOCATIONS, CONTRACT_NET),
    "picker_brain": choice("Picker decision making", PICKER_BRAINS, RULES),
    "path_planner": choice("Path planning", PATH_PLANNERS, INDEPENDENT),
    "drone_strategy": choice("Drone search strategy", DRONE_STRATEGIES, ROAM),
    "n_pickers": {"type": "SliderInt", "value": 4, "label": "Picker robots", "min": 1, "max": 10, "step": 1},
    "n_drones": {"type": "SliderInt", "value": 2, "label": "Explorer drones", "min": 1, "max": 6, "step": 1},
}

page = SolaraViz(
    FarmModel(seed=42),
    components=[
        FarmSpace,
        Status,
        make_plot_component(["Delivered", "Wasted", "Ripe on field"]),
        make_plot_component(["Mean picker battery", "Mean drone battery"]),
    ],
    model_params=model_params,
    name="Robot-Assisted Farming (Mesa 3)",
)
page  # noqa: B018 - Solara renders the module-level ``page``
