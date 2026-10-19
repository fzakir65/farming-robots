"""The farm model: builds the environment, creates the robots, runs the crop
life-cycle and collects the KPIs.
"""
import mesa

from .compat import MESA3, RandomScheduler
from .constants import (
    ALLOCATIONS,
    BASIC,
    BDI,
    BOLD,
    BROADCAST,
    CBS,
    CBS_MAX_NODES,
    CBS_WINDOW,
    CENTRAL_VRP,
    COMM_MODES,
    COMMITMENTS,
    CONTRACT_NET,
    DEFAULT_DROP_PROB,
    DEFAULT_MSG_DELAY,
    DRONE_STRATEGIES,
    FICTITIOUS,
    IDEAL,
    INDEPENDENT,
    LLM,
    LLM_MODEL,
    LOSSY,
    NETWORKS,
    NOVEL,
    PATH_PLANNERS,
    PICKER_BRAINS,
    RL,
    ROAM,
    RULES,
    STRATEGIC_MARKUP,
    XMPP,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
    FRUITING_PROB,
    MODES,
    PICKER_CAPACITY,
    REPLANT_JITTER,
    RIPE_AGE,
    ROT_AGE,
)
from .drone import ExplorerDrone
from .environment import (
    DRONE_PAD,
    EMPTY,
    PICKER_BASE,
    Base,
    FarmMap,
    Ground,
    Tree,
    Water,
    build_layout,
    patches_for_strip,
)
from .messages import LossyBus, MessageBus
from .picker import PickerRobot
from .sectors import FictitiousPlayer, make_sectors


# ---------------------------------------------------------------------- #
# KPI reporters (used by the DataCollector, the UI charts and experiments)
# ---------------------------------------------------------------------- #
def ripe_on_field(model):
    return sum(1 for t in model.trees if t.is_ripe)


def mean_picker_battery(model):
    return sum(p.battery for p in model.pickers) / max(len(model.pickers), 1)


def mean_drone_battery(model):
    return sum(d.battery for d in model.drones) / max(len(model.drones), 1)


def total_energy(model):
    return sum(a.energy_used for a in model.schedule.agents)


def energy_per_strawberry(model):
    """Energy spent by all robots per strawberry delivered (lower is better)."""
    return total_energy(model) / model.delivered if model.delivered else 0.0


def mean_pick_latency(model):
    """Average number of steps between a crop ripening and it being picked."""
    lat = model.pick_latencies
    return sum(lat) / len(lat) if lat else 0.0


def picker_utilisation(model):
    """Share of picker time spent working away from base."""
    steps = model.schedule.steps * max(len(model.pickers), 1)
    return sum(p.busy_steps for p in model.pickers) / steps if steps else 0.0


def harvest_rate(model):
    """Share of all strawberries that ripened which were picked rather than wasted."""
    total = model.picked + model.wasted
    return model.picked / total if total else 1.0


def mean_detect_latency(model):
    """Average steps between a crop ripening and a drone first seeing it."""
    lat = model.detect_latencies
    return sum(lat) / len(lat) if lat else 0.0


def blocked_steps(model):
    """Picker-steps spent waiting behind another picker (traffic in the lanes)."""
    return sum(p.blocked_total for p in model.pickers)


def mean_regret(model):
    """Auctions: how much more the winner's true cost was than the cheapest bidder's."""
    r = model.regrets
    return sum(r) / len(r) if r else 0.0


def overpayment(model):
    """Auctions: total paid to winners divided by their true costs (1 = no surplus)."""
    return model.total_payments / model.total_true_cost if model.total_true_cost else 0.0


class FarmModel(mesa.Model):
    """
    Robot-assisted strawberry farm.

    Parameters
    ----------
    mode : one of ``MODES`` (Basic, Extended, Novel)
    comm_mode : ``BROADCAST`` or ``DIRECT``, how drones message pickers in the extended mode
    n_pickers, n_drones : number of robots of each type, set by the operator
    picker_capacity : strawberries a picker can store before returning to base
    initial_ripe_patches : crop areas that start ripe
    random_layout : randomise the river position and the length of the tree rows
    max_steps : hard stop for the simulation
    seed : random seed, for reproducible experiments

    Advanced options (all default to the original behaviour):
    allocation : novel mode, one of ``ALLOCATIONS`` (auctions, central VRP, LLM)
    strategic_bidders : share of pickers that shade their bids in first-price auctions
    picker_brain : ``RULES``, ``BDI`` (AgentSpeak) or ``RL`` (learned task choice)
    bdi_commitment : ``BOLD`` or ``CAUTIOUS`` (BDI pickers only)
    rl_policy : a policy callable or a path to a MaskablePPO .zip (default models/)
    path_planner : ``INDEPENDENT`` (Dijkstra per robot) or ``CBS`` (windowed CBS)
    drone_strategy : ``ROAM``, ``NASH`` (sector game equilibrium) or ``FICTITIOUS``
    network : ``IDEAL``, ``LOSSY`` (drop_prob, msg_delay) or ``XMPP`` (SPADE)
    llm_model, llm_client : Ollama model name / injected client (tests)
    """

    def __init__(
        self,
        mode=BASIC,
        comm_mode=BROADCAST,
        n_pickers=4,
        n_drones=2,
        picker_capacity=PICKER_CAPACITY,
        initial_ripe_patches=8,
        random_layout=False,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        max_steps=DEFAULT_MAX_STEPS,
        seed=None,
        allocation=CONTRACT_NET,
        strategic_bidders=0.0,
        picker_brain=RULES,
        bdi_commitment=BOLD,
        rl_policy=None,
        path_planner=INDEPENDENT,
        drone_strategy=ROAM,
        network=IDEAL,
        drop_prob=DEFAULT_DROP_PROB,
        msg_delay=DEFAULT_MSG_DELAY,
        llm_model=LLM_MODEL,
        llm_client=None,
    ):
        super().__init__(seed=seed)
        for value, allowed, name in (
            (mode, MODES, "mode"), (comm_mode, COMM_MODES, "comm_mode"),
            (allocation, ALLOCATIONS, "allocation"), (picker_brain, PICKER_BRAINS, "picker_brain"),
            (bdi_commitment, COMMITMENTS, "bdi_commitment"), (path_planner, PATH_PLANNERS, "path_planner"),
            (drone_strategy, DRONE_STRATEGIES, "drone_strategy"), (network, NETWORKS, "network"),
        ):
            if value not in allowed:
                raise ValueError(f"Unknown {name} {value!r}; expected one of {allowed}")
        self.mode = mode
        self.comm_mode = comm_mode
        self.allocation = allocation
        self.picker_brain = picker_brain
        self.path_planner = path_planner
        self.drone_strategy = drone_strategy
        self.network = network
        self.crops_age = mode != BASIC          # the environment ages crops after Basic
        self.max_steps = max_steps
        self._next_uid = 0

        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        self.schedule = RandomScheduler(self)
        if network == LOSSY:
            self.bus = LossyBus(self, drop_prob, msg_delay)
        elif network == XMPP:
            from .xmpp_bus import XmppBus

            self.bus = XmppBus(self)
        else:
            self.bus = MessageBus(self)

        # KPI counters
        self.delivered = 0      # strawberries unloaded at base
        self.picked = 0         # strawberries cut from trees
        self.wasted = 0         # strawberries that rotted before being picked
        self.replanted = 0      # crop areas re-seeded
        self.pick_latencies = []
        self.detect_latencies = []
        self.contracts_awarded = 0
        self.contracts_completed = 0
        self.total_payments = 0.0
        self.total_true_cost = 0.0
        self.regrets = []
        self.english_rounds = 0
        self.wdp_solves = 0
        self.llm_fallbacks = 0
        self.sector_choices = []

        # Shared coordination state
        self.report_log = {}        # patch_id -> step it was last announced
        self.guarded_patches = {}   # basic: patch_id -> drone guarding it
        self._task_counter = 0

        self.build_environment(initial_ripe_patches, random_layout)
        self.create_robots(n_pickers, n_drones, picker_capacity)
        self.setup_extensions(strategic_bidders, bdi_commitment, rl_policy, llm_model, llm_client)

        reporters = dict(
            model_reporters={
                "Delivered": lambda m: m.delivered,
                "Picked": lambda m: m.picked,
                "Wasted": lambda m: m.wasted,
                "Ripe on field": ripe_on_field,
                "Mean picker battery": mean_picker_battery,
                "Mean drone battery": mean_drone_battery,
                "Energy used": total_energy,
                "Energy per strawberry": energy_per_strawberry,
                "Mean pick latency": mean_pick_latency,
                "Picker utilisation": picker_utilisation,
                "Harvest rate": harvest_rate,
                "Messages sent": lambda m: m.bus.sent,
                "Mean detection latency": mean_detect_latency,
                "Blocked steps": blocked_steps,
                "Mean regret": mean_regret,
                "Overpayment": overpayment,
                "Messages dropped": lambda m: getattr(m.bus, "dropped", 0),
            },
        )
        robot_reporters = {
            "State": lambda a: a.state,
            "Battery": lambda a: a.battery,
            "Load": lambda a: getattr(a, "load", None),
        }
        if MESA3:
            # Mesa 3 collects agent data from every agent, including ~1000 static cells
            reporters["agenttype_reporters"] = {PickerRobot: robot_reporters, ExplorerDrone: robot_reporters}
        else:
            reporters["agent_reporters"] = {"Type": lambda a: type(a).__name__, **robot_reporters}
        self.datacollector = mesa.DataCollector(**reporters)
        self.running = True
        self.datacollector.collect(self)

    # ------------------------------------------------------------------ #
    # Set-up
    # ------------------------------------------------------------------ #
    def build_environment(self, initial_ripe_patches, random_layout):
        width, height = self.grid.width, self.grid.height
        layout = build_layout(width, height, self.random, random_layout)

        self.trees = []
        self.trees_by_pos = {}
        self.patches = {}
        for strip in layout["strips"]:
            for cells in patches_for_strip(*strip):
                patch_id = len(self.patches)
                self.patches[patch_id] = []
                for cell in cells:
                    tree = Tree(self.next_id(), self, patch_id)
                    self.grid.place_agent(tree, cell)
                    self.trees.append(tree)
                    self.trees_by_pos[cell] = tree
                    self.patches[patch_id].append(tree)

        water_cells = [(x, y) for x in layout["river"] for y in range(height)]
        for cell in water_cells:
            self.grid.place_agent(Water(self.next_id(), self), cell)
        for cell in layout["picker_base"]:
            self.grid.place_agent(Base(self.next_id(), self, PICKER_BASE), cell)
        for cell in layout["drone_pad"]:
            self.grid.place_agent(Base(self.next_id(), self, DRONE_PAD), cell)
        special = set(water_cells) | set(layout["picker_base"]) | set(layout["drone_pad"])
        for x in range(width):
            for y in range(height):
                if (x, y) not in self.trees_by_pos and (x, y) not in special:
                    self.grid.place_agent(Ground(self.next_id(), self), (x, y))

        self.farm = FarmMap(
            width, height, self.trees_by_pos.keys(), water_cells,
            layout["picker_base"], layout["drone_pad"],
        )
        self.seed_crops(initial_ripe_patches)

    def seed_crops(self, initial_ripe_patches):
        """Make some crop areas ripe. With ageing on, give the rest a random age."""
        patch_ids = list(self.patches)
        self.random.shuffle(patch_ids)
        ripe = set(patch_ids[:initial_ripe_patches])
        for patch_id, trees in self.patches.items():
            if patch_id in ripe:
                age = RIPE_AGE + (self.random.randint(0, (ROT_AGE - RIPE_AGE) // 2) if self.crops_age else 0)
                for t in trees:
                    t.ripen()
                    t.age = age
            elif self.crops_age:
                age = self.random.randint(0, RIPE_AGE - 1)
                for t in trees:
                    t.replant(delay=-age, bears_fruit=self.random.random() < FRUITING_PROB)

    def create_robots(self, n_pickers, n_drones, picker_capacity):
        self.pickers = []
        self.drones = []
        for i in range(n_pickers):
            picker = PickerRobot(self.next_id(), self, picker_capacity)
            self.add_robot(picker, self.farm.picker_base[i % len(self.farm.picker_base)])
            self.pickers.append(picker)
        for i in range(n_drones):
            drone = ExplorerDrone(self.next_id(), self)
            self.add_robot(drone, self.farm.drone_pad[i % len(self.farm.drone_pad)])
            self.drones.append(drone)

    def add_robot(self, robot, pos):
        self.grid.place_agent(robot, pos)
        self.schedule.add(robot)
        self.bus.register(robot.unique_id)

    def next_id(self):
        """Our own ids (Mesa 3 removed ``Model.next_id``)."""
        self._next_uid += 1
        return self._next_uid

    def setup_extensions(self, strategic_bidders, bdi_commitment, rl_policy, llm_model, llm_client):
        """Create the optional pieces: brains, bidders, planners, sector game, LLM."""
        # self-interested bidders (first-price auctions)
        n_strategic = round(strategic_bidders * len(self.pickers))
        for p in self.pickers[:n_strategic]:
            p.markup = self.random.uniform(*STRATEGIC_MARKUP)
        # picker brains
        if self.picker_brain == BDI:
            from .bdi import BDIBrain

            for p in self.pickers:
                p.brain = BDIBrain(p, bdi_commitment)
        elif self.picker_brain == RL:
            from .rl import RLBrain, SB3Policy, load_policy

            if callable(rl_policy):
                policy = rl_policy
            elif rl_policy:
                policy = SB3Policy(rl_policy)
            else:
                policy = load_policy()
            for p in self.pickers:
                p.brain = RLBrain(policy)
        # multi-robot path planning
        self.paths = None
        if self.path_planner == CBS:
            from .mapf import PathCoordinator

            self.paths = PathCoordinator(self, CBS_WINDOW, CBS_MAX_NODES)
        # central planner
        self.planner = None
        if self.mode == NOVEL and self.allocation == CENTRAL_VRP:
            from .planner import CentralPlanner

            self.planner = CentralPlanner(self, self.next_id())
        # LLM coordinator
        self.llm = None
        if self.mode == NOVEL and self.allocation == LLM:
            from .llm import OllamaCoordinator

            self.llm = OllamaCoordinator(llm_model, client=llm_client)
        # drones' sector game
        self.sectors = make_sectors(self.farm) if self.drone_strategy != ROAM else []
        total = max(len(self.trees), 1)
        self.sector_values = [
            2.0 * sum(1 for t in self.trees if x0 <= t.pos[0] <= x1 and y0 <= t.pos[1] <= y1) / total
            for (x0, x1, y0, y1) in self.sectors
        ]
        if self.drone_strategy == FICTITIOUS:
            for d in self.drones:
                d.player = FictitiousPlayer(len(self.sectors), self.sector_values)

    def close(self):
        """Release network resources (needed for the XMPP bus)."""
        self.bus.close()

    # ------------------------------------------------------------------ #
    # Queries used by the agents
    # ------------------------------------------------------------------ #
    def tree_at(self, pos):
        return self.trees_by_pos.get(pos)

    def ripe_trees_near(self, pos, radius):
        x, y = pos
        found = []
        for tx in range(x - radius, x + radius + 1):
            for ty in range(y - radius, y + radius + 1):
                tree = self.trees_by_pos.get((tx, ty))
                if tree is not None and tree.is_ripe:
                    found.append(tree)
        return found

    def picker_at(self, pos):
        return any(isinstance(a, PickerRobot) for a in self.grid.get_cell_list_contents([pos]))

    def new_task_id(self):
        self._task_counter += 1
        return self._task_counter

    def record_pick(self, tree):
        self.picked += 1
        if tree.ripe_since is not None:
            self.pick_latencies.append(self.schedule.steps - tree.ripe_since)

    def record_detection(self, tree):
        if tree.ripe_since is not None:
            self.detect_latencies.append(self.schedule.steps - tree.ripe_since)

    def record_award(self, picker_id, payment, true_cost, regret):
        """Book-keeping for every contract: the payment, true cost and regret."""
        self.contracts_awarded += 1
        if payment is not None and true_cost is not None:
            self.total_payments += payment
            self.total_true_cost += true_cost
            for p in self.pickers:
                if p.unique_id == picker_id:
                    p.earnings += payment
        if regret is not None:
            self.regrets.append(regret)

    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #
    def grow_crops(self):
        """The environment ages every crop and re-seeds areas that are fully picked."""
        for tree in self.trees:
            tree.grow()
        for trees in self.patches.values():
            if all(t.state == EMPTY for t in trees):
                delay = self.random.randint(0, REPLANT_JITTER)
                for t in trees:
                    t.replant(delay, bears_fruit=self.random.random() < FRUITING_PROB)
                self.replanted += 1

    def step(self):
        if self.crops_age:
            self.grow_crops()
        if self.planner is not None:
            self.planner.step()
        if self.paths is not None:
            self.paths.before_step()
        self.schedule.step()
        self.bus.end_step()
        self.datacollector.collect(self)
        if self.is_finished():
            self.running = False

    def is_finished(self):
        if self.schedule.steps >= self.max_steps:
            return True
        if not any(p.is_active for p in self.pickers):
            return True
        if self.mode == BASIC:
            # Basic mission is complete once every ripe strawberry is at the base
            return ripe_on_field(self) == 0 and all(p.load == 0 for p in self.pickers)
        return False
