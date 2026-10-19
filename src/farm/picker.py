"""Amphibious picker robot.

Hardware capabilities (from the brief) and how they are modelled:
  * cuts strawberries with an arm   -> ``pick`` reaches trees within ARM_REACH
  * stores and carries them         -> ``load`` up to ``capacity``; ``unload`` at base
  * cannot pass through trees       -> only moves on lanes (``FarmMap.passable``)
  * crosses water slowly            -> needs WATER_SLOWDOWN steps to leave a water cell
  * battery drains as time passes   -> ``use_energy``; ``charge`` at base

Decision making follows the worksheet pattern: ``make_decision`` returns the name of
an action method, which ``step`` then runs. Which reasoning produces that name is
pluggable (``model.picker_brain``):
  * rule-based (event -> condition -> action), ``rule_based_decision`` below;
  * BDI, an AgentSpeak program (``farm/bdi.py``, ``farm/asl/picker.asl``);
  * learned, a MaskablePPO policy choosing the next crop (``farm/rl.py``).
The statechart lives in ``docs/design.md``.
"""
import itertools

from . import messages as msg
from .compat import FarmAgent
from .constants import (
    ARM_REACH,
    BASIC,
    CBS,
    COMBINATORIAL,
    CONTRACT_NET,
    MAX_BUNDLE_SIZE,
    EXPLORE_PATIENCE,
    NOVEL,
    PICKER_BATTERY,
    PICKER_CAPACITY,
    PICKER_CHARGE_RATE,
    PICKER_IDLE_COST,
    PICKER_MOVE_COST,
    PICKER_PICK_COST,
    PICKER_RESERVE,
    RETURN_MARGIN,
    SENSOR_RANGE,
    URGENCY_WEIGHT,
    WATER_SLOWDOWN,
)

# Picker states
AT_BASE = "at_base"
CHARGING = "charging"
EXPLORING = "exploring"
MOVING_TO_CROP = "moving_to_crop"
PICKING = "picking"
RETURNING = "returning"
OUT_OF_BATTERY = "out_of_battery"

QUEUE_PENALTY = 10      # novel: extra bid cost for each task already committed to
BLOCKED_PATIENCE = 3    # steps to wait behind another robot before planning around it
STANDOFF_LIMIT = 8      # steps without progress after which a robot squeezes past others
DETOUR_LIMIT = 6        # a way round another robot may be at most this many steps longer


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class PickerRobot(FarmAgent):
    """An autonomous amphibious strawberry-picking robot."""

    def __init__(self, unique_id, model, capacity=PICKER_CAPACITY):
        super().__init__(unique_id, model)
        self.capacity = capacity
        self.load = 0
        self.battery = PICKER_BATTERY
        self.needs_recharge = False
        self.state = AT_BASE
        # navigation
        self.path = []          # cells still to visit
        self.goals = set()      # cells that count as "arrived"
        self.target = None      # tree being approached
        self.heading = (0, 1)
        self.water_progress = 0
        self.blocked_steps = 0
        self.planned_goals = set()
        self.best_remaining = float("inf")
        self.no_progress = 0
        self.fruitless_steps = 0
        # known crop locations: tree pos -> {"urgency", "task_id", "manager"}
        self.tasks = {}
        self.chosen_task = None     # set by a BDI or learned brain before "go_to_chosen_task"
        self.brain = None           # None = rule-based; otherwise has decide(picker)
        self.markup = 0.0           # > 0 for a self-interested bidder (first-price auctions)
        # statistics (KPIs)
        self.earnings = 0.0
        self.blocked_total = 0
        self.delivered = 0
        self.energy_used = 0.0
        self.distance = 0
        self.busy_steps = 0

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def at_base(self):
        return self.pos in self.model.farm.picker_base

    @property
    def is_full(self):
        return self.load >= self.capacity

    @property
    def is_active(self):
        return self.state != OUT_OF_BATTERY

    @property
    def spare_capacity(self):
        """Free storage slots not already promised to known tasks."""
        return self.capacity - self.load - len(self.tasks)

    # ------------------------------------------------------------------ #
    # Main loop: perceive -> decide -> act
    # ------------------------------------------------------------------ #
    def step(self):
        if self.state == OUT_OF_BATTERY:
            return
        self.handle_messages()
        self.observe()
        action = self.make_decision()
        getattr(self, action)()
        if not self.at_base:
            self.busy_steps += 1
            self.use_energy(PICKER_IDLE_COST)

    def make_decision(self):
        """Ask the robot's brain for the next action (rule-based by default)."""
        if self.brain is not None:
            return self.brain.decide(self)
        return self.rule_based_decision()

    def rule_based_decision(self):
        """
        Rule-based decision model. Rules are checked in priority order and the
        first that matches gives the action to execute.
        """
        if self.at_base:
            if self.load > 0:
                return "unload"
            if self.battery < PICKER_BATTERY:
                return "charge"
            if self.model.mode == BASIC:
                return "explore"
            if self.tasks:
                return "go_to_next_task"
            return "wait_at_base"

        # Safety first: full storage or not enough energy to get home
        if self.should_return():
            return "return_to_base"
        # Opportunistic picking: anything ripe within reach of the arm
        if not self.is_full and self.ripe_in_reach():
            return "pick"
        if self.state == MOVING_TO_CROP and self.path:
            return "move_to_crop"

        if self.model.mode == BASIC:
            if self.ripe_in_sight():
                return "approach_crop"
            if self.state == RETURNING:
                return "return_to_base"
            if self.load > 0 and self.fruitless_steps >= EXPLORE_PATIENCE:
                return "return_to_base"
            return "explore"

        # Extended / novel: work only on crops the drones told us about
        if self.tasks:
            return "go_to_next_task"
        return "return_to_base"

    # ------------------------------------------------------------------ #
    # Perception
    # ------------------------------------------------------------------ #
    def observe(self):
        """Look around and forget tasks whose strawberries are visibly gone."""
        for pos in list(self.tasks):
            if chebyshev(self.pos, pos) <= SENSOR_RANGE and not self.model.tree_at(pos).is_ripe:
                self.finish_task_tree(pos)
        if (
            self.target is not None
            and chebyshev(self.pos, self.target) <= SENSOR_RANGE
            and not self.model.tree_at(self.target).is_ripe
        ):
            self.clear_target()

    def ripe_in_reach(self):
        return self.model.ripe_trees_near(self.pos, ARM_REACH)

    def ripe_in_sight(self):
        return self.model.ripe_trees_near(self.pos, SENSOR_RANGE)

    def should_return(self):
        """
        Full storage or low battery. A low-battery decision is latched until the robot
        has recharged: without it, the threshold falls as the robot nears home and it
        would turn back to work, ping-ponging until the battery runs out.
        """
        if self.battery <= self.energy_to_base() + PICKER_RESERVE:
            self.needs_recharge = True
        return self.is_full or self.needs_recharge

    def energy_to_base(self):
        """Energy needed to drive home from here, with a margin for traffic."""
        return self.trip_energy(self.model.farm.base_distance.get(self.pos, 0))

    # ------------------------------------------------------------------ #
    # Communication
    # ------------------------------------------------------------------ #
    def handle_messages(self):
        for m in self.model.bus.receive(self.unique_id):
            if m.performative == msg.INFORM_CROP:
                self.learn_task(m.content, task_id=None, manager=m.sender)
            elif m.performative == msg.CFP and "areas" in m.content:
                self.respond_to_bundle_cfp(m)
            elif m.performative == msg.CFP:
                self.respond_to_cfp(m)
            elif m.performative == msg.ACCEPT:
                self.learn_task(m.content, task_id=m.content["task_id"], manager=m.sender)
            # REJECT needs no action: the robot simply did not win the task

    def learn_task(self, content, task_id, manager):
        for pos in content["trees"]:
            pos = tuple(pos)
            known = self.tasks.get(pos)
            if known is None or task_id is not None:
                self.tasks[pos] = {
                    "urgency": content["urgency"],
                    "task_id": task_id,
                    "manager": manager,
                    "order": content.get("order"),
                }
            else:
                known["urgency"] = max(known["urgency"], content["urgency"])

    def respond_to_cfp(self, cfp):
        """Contract Net: answer a call for proposals with a bid (cost) or a refusal."""
        cost = self.estimate_cost(cfp.content)
        if cost is None:
            self.model.bus.send(msg.REFUSE, self.unique_id, cfp.sender, {"task_id": cfp.content["task_id"]})
        else:
            self.model.bus.send(
                msg.PROPOSE, self.unique_id, cfp.sender,
                {"task_id": cfp.content["task_id"], "cost": self.bid_for(cost), "true_cost": cost},
            )

    def bid_for(self, cost):
        """
        The price asked for a job. When the winner is paid its own bid (first-price
        auctions), a self-interested robot shades its bid upwards. In a Vickrey or
        English auction, bidding the true cost is a dominant strategy.
        """
        if self.markup and self.model.allocation in (CONTRACT_NET, COMBINATORIAL):
            return round(cost * (1 + self.markup), 2)
        return cost

    def respond_to_bundle_cfp(self, cfp):
        """Combinatorial auction: bid for every bundle of up to MAX_BUNDLE_SIZE areas."""
        bundles = self.bundle_bids(cfp.content["areas"])
        if not bundles:
            self.model.bus.send(msg.REFUSE, self.unique_id, cfp.sender, {"task_id": cfp.content["task_id"]})
            return
        self.model.bus.send(msg.PROPOSE, self.unique_id, cfp.sender,
                            {"task_id": cfp.content["task_id"], "bundles": bundles})

    def bundle_bids(self, areas):
        """
        One bid per affordable bundle. The cost is the shortest tour through the
        bundle's areas (all visiting orders are tried), plus the usual penalties.
        """
        if not self.is_active or self.spare_capacity <= 0:
            return []
        farm = self.model.farm
        start = list(self.tasks)[-1] if self.tasks else self.pos
        if start not in farm.passable:
            start = min(farm.pick_positions(start), key=lambda p: farm.base_distance.get(p, 1e9))
        stops = {}
        for a in areas:
            cells = [c for t in a["trees"] for c in farm.pick_positions(tuple(t))]
            if cells:
                cell = min(cells, key=lambda c: (farm.base_distance.get(c, 1e9), c))
                stops[a["patch"]] = (cell, len(a["trees"]))
        maps = {start: farm.distance_map([start])}
        for cell, _ in stops.values():
            maps[cell] = farm.distance_map([cell])
        bids = []
        for size in range(1, min(MAX_BUNDLE_SIZE, len(stops)) + 1):
            for combo in itertools.combinations(sorted(stops), size):
                trees = sum(stops[a][1] for a in combo)
                if trees > self.spare_capacity:
                    continue
                best = None
                for order in itertools.permutations(combo):
                    here, length = start, 0
                    for a in order:
                        cell = stops[a][0]
                        length += maps[here].get(cell, 1e9)
                        here = cell
                    if best is None or length < best[0]:
                        best = (length, here)
                length, last = best
                if length >= 1e9:
                    continue
                cost = length + QUEUE_PENALTY * len(self.tasks)
                if self.battery - self.trip_energy(length + farm.base_distance.get(last, 0)) < PICKER_RESERVE:
                    to_base = 0 if self.at_base else farm.base_distance.get(self.pos, 0)
                    cost += to_base + (PICKER_BATTERY - self.battery) / PICKER_CHARGE_RATE
                cost = round(cost, 2)
                bids.append({"areas": list(combo), "trees": trees, "bid": self.bid_for(cost), "cost": cost})
        return bids

    def estimate_cost(self, task):
        """
        Bid for a task: the estimated number of steps before this robot could start
        picking it, given what it has already promised to do. Returns None when the
        robot cannot take the task (out of battery, no storage left, unreachable).
        """
        if not self.is_active or self.spare_capacity <= 0:
            return None
        farm = self.model.farm
        # Work is queued, so travel starts from the last committed crop (or from here)
        start = list(self.tasks)[-1] if self.tasks else self.pos
        if start not in farm.passable:
            start = min(farm.pick_positions(start), key=lambda p: farm.base_distance.get(p, 1e9))
        dist = farm.distance_map([start])
        tree = tuple(task["trees"][0])
        reach = [dist[p] for p in farm.pick_positions(tree) if p in dist]
        if not reach:
            return None
        travel = min(reach)
        home = min(farm.base_distance.get(p, 1e9) for p in farm.pick_positions(tree))
        cost = travel + QUEUE_PENALTY * len(self.tasks)
        energy_needed = self.trip_energy(travel + home)
        if self.battery - energy_needed < PICKER_RESERVE:
            # It would have to go home and recharge first
            to_base = 0 if self.at_base else farm.base_distance.get(self.pos, 0)
            cost += to_base + (PICKER_BATTERY - self.battery) / PICKER_CHARGE_RATE
        if self.is_full:
            cost += farm.base_distance.get(self.pos, 0)
        return round(cost, 2)

    def finish_task_tree(self, pos):
        """Drop a task tree; when a contracted task is complete, tell its manager."""
        info = self.tasks.pop(pos, None)
        if info is None:
            return
        task_id = info["task_id"]
        if task_id is not None and not any(t["task_id"] == task_id for t in self.tasks.values()):
            self.model.bus.send(msg.INFORM_DONE, self.unique_id, info["manager"], {"task_id": task_id})

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def wait_at_base(self):
        """Stay docked at the base until a drone signals crops to collect."""
        self.state = AT_BASE

    def charge(self):
        self.state = CHARGING
        self.battery = min(PICKER_BATTERY, self.battery + PICKER_CHARGE_RATE)
        if self.battery >= PICKER_BATTERY:
            self.needs_recharge = False

    def unload(self):
        """Hand the strawberries over at the base station."""
        self.model.delivered += self.load
        self.delivered += self.load
        self.load = 0
        self.fruitless_steps = 0
        self.clear_target()
        self.state = AT_BASE

    def pick(self):
        """Cut the most urgent ripe strawberries within reach and store them."""
        self.state = PICKING
        tree = max(self.ripe_in_reach(), key=lambda t: t.urgency)
        if tree.harvest():
            self.load += 1
            self.fruitless_steps = 0
            self.use_energy(PICKER_PICK_COST)
        if tree.pos in self.tasks:
            self.finish_task_tree(tree.pos)
        if tree.pos == self.target:
            self.clear_target()

    def explore(self):
        """Basic mode: random walk with momentum along the lanes, looking for crops."""
        self.state = EXPLORING
        self.fruitless_steps += 1
        options = [n for n in self.model.farm.neighbours(self.pos) if not self.blocked_by_picker(n)]
        if not options:
            return
        forward = (self.pos[0] + self.heading[0], self.pos[1] + self.heading[1])
        if forward in options and self.random.random() < 0.8:
            nxt = forward
        else:
            back = (self.pos[0] - self.heading[0], self.pos[1] - self.heading[1])
            nxt = self.random.choice([o for o in options if o != back] or options)
        self.heading = (nxt[0] - self.pos[0], nxt[1] - self.pos[1])
        self.step_to(nxt)

    def approach_crop(self):
        """Basic mode: head for the closest ripe tree the robot can see."""
        tree = min(self.ripe_in_sight(), key=lambda t: chebyshev(self.pos, t.pos))
        if self.set_target(tree.pos):
            self.follow_path()
        else:
            self.explore()

    def task_score(self, pos, dist=None):
        """Lower is better: path length minus a bonus for urgent crops."""
        farm = self.model.farm
        dist = dist if dist is not None else farm.distance_map([self.pos])
        reach = [dist[p] for p in farm.pick_positions(pos) if p in dist]
        if not reach:
            return float("inf")
        return min(reach) - URGENCY_WEIGHT * self.tasks[pos]["urgency"]

    def choose_task(self):
        """The best known task. Tasks routed by the central planner keep their order."""
        if not self.tasks:
            return None
        ordered = [p for p, info in self.tasks.items() if info.get("order") is not None]
        if ordered:
            return min(ordered, key=lambda p: (self.tasks[p]["order"], p))
        dist = self.model.farm.distance_map([self.pos])
        return min(self.tasks, key=lambda p: self.task_score(p, dist))

    def go_to_next_task(self):
        """Choose the best known task (near and urgent) and start driving to it."""
        self.head_for(self.choose_task())

    def go_to_chosen_task(self):
        """Drive to the task picked by a BDI or learned brain."""
        pos, self.chosen_task = self.chosen_task, None
        if pos in self.tasks:
            self.head_for(pos)
        elif self.tasks:
            self.go_to_next_task()
        else:
            self.return_to_base()

    def head_for(self, best):
        """Set off for task ``best`` if the battery covers the round trip."""
        farm = self.model.farm
        dist = farm.distance_map([self.pos])
        reach = [dist[p] for p in farm.pick_positions(best) if p in dist]
        if not reach:
            self.finish_task_tree(best)       # unreachable: give it up
            return
        # Only set off if the battery covers the trip there and back again
        home = min(farm.base_distance.get(p, 0) for p in farm.pick_positions(best))
        if self.battery < self.trip_energy(min(reach) + home) + PICKER_RESERVE:
            self.needs_recharge = True
            self.return_to_base()
            return
        if not self.set_target(best):
            self.finish_task_tree(best)
            return
        self.follow_path()

    @staticmethod
    def trip_energy(steps):
        """Energy for a trip of ``steps`` steps, with the margin for traffic."""
        return steps * (PICKER_MOVE_COST + PICKER_IDLE_COST) * RETURN_MARGIN

    def move_to_crop(self):
        self.state = MOVING_TO_CROP
        self.follow_path()

    def return_to_base(self):
        """Drive back to the base station (to unload, recharge or wait)."""
        base = set(self.model.farm.picker_base)
        if self.state != RETURNING or self.goals != base or not self.path:
            self.state = RETURNING
            self.target = None
            self.goals = base
            self.plan()
        self.follow_path()

    # ------------------------------------------------------------------ #
    # Movement helpers
    # ------------------------------------------------------------------ #
    def set_target(self, tree_pos):
        """Plan a route to a cell from which ``tree_pos`` is within reach."""
        self.target = tree_pos
        self.goals = set(self.model.farm.pick_positions(tree_pos))
        self.state = MOVING_TO_CROP
        if not self.plan():
            self.clear_target()
            return False
        return True

    def clear_target(self):
        self.target = None
        self.path = []
        self.goals = set()

    def plan(self, avoid_pickers=False):
        """Compute ``self.path`` to ``self.goals``. Returns False if unreachable."""
        blocked = frozenset()
        if avoid_pickers:
            blocked = frozenset(
                p.pos for p in self.model.pickers
                if p is not self and p.pos not in self.model.farm.picker_base
            ) - self.goals
        path = self.model.farm.shortest_path(self.pos, self.goals, blocked)
        if avoid_pickers and (path is None or len(path) > len(self.path) + DETOUR_LIMIT):
            return False    # no sensible way round: keep the old route and wait instead
        self.path = path or []
        if self.model.paths is not None and path:
            self.model.paths.request(self)       # CBS: plan jointly at the next step
        if self.goals != self.planned_goals:     # new destination: restart progress tracking
            self.planned_goals = set(self.goals)
            self.best_remaining = float("inf")
            self.no_progress = 0
        return path is not None

    def follow_path(self):
        """
        Take the next step of the route, with collision avoidance:
          1. if another picker is in the way, wait;
          2. every BLOCKED_PATIENCE steps, try to plan a route around it;
          3. if the route has not got shorter for STANDOFF_LIMIT steps (deadlock,
             or two robots dodging each other back and forth), squeeze past.
        """
        if not self.path:
            return
        stuck = self.no_progress >= STANDOFF_LIMIT
        if stuck and len(self.path) > self.best_remaining:
            self.plan()     # abandon a detour that is not paying off: take the direct route
        nxt = self.path[0]
        if nxt == self.pos:                     # a planned wait (CBS)
            self.path.pop(0)
            if self.model.farm.is_water(self.pos):
                self.water_progress += 1
        elif self.blocked_by_picker(nxt) and not stuck:
            self.blocked_steps += 1
            self.blocked_total += 1
            if self.model.path_planner == CBS:
                self.model.paths.request(self)
            elif self.blocked_steps % BLOCKED_PATIENCE == 0:
                self.plan(avoid_pickers=True)
        elif self.step_to(nxt) and self.path:   # the path is cleared if the battery just died
            self.path.pop(0)
            self.blocked_steps = 0
        self.track_progress()

    def track_progress(self):
        remaining = len(self.path)
        if remaining < self.best_remaining:
            self.best_remaining = remaining
            self.no_progress = 0
        else:
            self.no_progress += 1

    def blocked_by_picker(self, pos):
        """Only one picker per cell, except at the base station."""
        return pos not in self.model.farm.picker_base and self.model.picker_at(pos)

    def step_to(self, nxt):
        """Move to an adjacent cell. Leaving water takes WATER_SLOWDOWN steps."""
        if self.model.farm.is_water(self.pos):
            self.water_progress += 1
            if self.water_progress < WATER_SLOWDOWN:
                return False
        self.water_progress = 0
        self.model.grid.move_agent(self, nxt)
        self.distance += 1
        self.use_energy(PICKER_MOVE_COST)
        return True

    def use_energy(self, amount):
        self.battery -= amount
        self.energy_used += amount
        if self.battery <= 0:
            self.battery = 0
            self.state = OUT_OF_BATTERY
            self.clear_target()
