"""Explorer drone.

Hardware capabilities (from the brief) and how they are modelled:
  * moves in 6 directions        -> move_front/back/left/right, ascend, descend
  * flies over trees             -> may enter a tree cell only above TREE_HEIGHT
  * takes pictures of the crops  -> ``scan``: the camera sees a radius equal to the altitude
  * computer vision -> urgency   -> ``Tree.urgency`` (how close the crop is to rotting)
  * signals other robots         -> sends messages over ``model.bus``

What the drone does with what it finds depends on the operation mode:
  * Basic:    fly to the crop and hover over it until a picker arrives
  * Extended: send the crop location to all pickers (broadcast) or to one (direct)
  * Novel:    allocate each crop area with the chosen method (``model.allocation``):
              a Contract Net / Vickrey / English / combinatorial auction where the
              drone is the auctioneer, a report to the central OR-Tools planner, or a
              call to a local LLM (Ollama).

Where the drone searches is a strategy too (``model.drone_strategy``): random
roaming, the Nash equilibrium of the sector congestion game, or fictitious play.
See ``farm/sectors.py``.
"""
from . import messages as msg
from .auctions import english_clock, first_price, regret, second_price, solve_winner_determination
from .compat import FarmAgent
from .constants import (
    ALTITUDE_CHANGE_PROB,
    AREA_VALUE,
    ARM_REACH,
    BASIC,
    BROADCAST,
    BUNDLE_WINDOW,
    CENTRAL_VRP,
    CFP_DEADLINE,
    COMBINATORIAL,
    ENGLISH,
    ENGLISH_DECREMENT,
    ENGLISH_START_PRICE,
    LLM,
    MAX_BUNDLE_AREAS,
    NASH,
    ROAM,
    SECTOR_LEARNING_RATE,
    SECTOR_PERIOD,
    VICKREY,
    CRUISE_ALTITUDE,
    DRONE_ALTITUDE_COST,
    DRONE_BATTERY,
    DRONE_CHARGE_RATE,
    DRONE_HOVER_COST,
    DRONE_MAX_WAIT,
    DRONE_MOVE_COST,
    DRONE_RESERVE,
    EXTENDED,
    MAX_ALTITUDE,
    NOVEL,
    REREPORT_INTERVAL,
    TREE_HEIGHT,
)
from .sectors import nash_assignment, sector_centre, sector_of

# Drone states
LANDED = "landed"
EXPLORING = "exploring"
GOING_TO_CROP = "going_to_crop"
WAITING = "waiting"
RETURNING = "returning"
OUT_OF_BATTERY = "out_of_battery"

# The four horizontal directions (the other two are up and down)
FRONT, BACK, LEFT, RIGHT = (0, 1), (0, -1), (-1, 0), (1, 0)
HORIZONTAL = [FRONT, BACK, LEFT, RIGHT]


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class ExplorerDrone(FarmAgent):
    """An autonomous drone that finds ripe strawberries and tells the pickers."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.altitude = 0
        self.battery = DRONE_BATTERY
        self.state = LANDED
        self.heading = self.random.choice(HORIZONTAL)
        self.target = None          # basic: tree the drone is guarding
        self.wait_steps = 0
        self.findings = []
        # novel mode: open calls for proposals, task_id -> details
        self.pending_cfps = {}
        self.pool = {}              # combinatorial / LLM: patch id -> (trees, step first seen)
        # sector game
        self.sector = None
        self.sector_since = 0
        self.player = None          # fictitious play beliefs, created by the model
        # statistics (KPIs)
        self.reports = 0
        self.energy_used = 0.0
        self.distance = 0

    @property
    def at_pad(self):
        return self.pos in self.model.farm.drone_pad

    @property
    def is_active(self):
        return self.state != OUT_OF_BATTERY

    # ------------------------------------------------------------------ #
    # Main loop: communicate -> perceive -> decide -> act
    # ------------------------------------------------------------------ #
    def step(self):
        if self.state == OUT_OF_BATTERY:
            return
        self.handle_messages()
        if self.model.mode == NOVEL:
            self.award_contracts()

        self.findings = self.scan()
        if self.findings and self.state == EXPLORING:
            if self.model.mode == EXTENDED:
                self.report_crops(self.findings)
            elif self.model.mode == NOVEL:
                self.allocate(self.findings)
        if self.model.mode == NOVEL and self.pool:
            self.flush_pool()

        action = self.make_decision()
        getattr(self, action)()
        if self.altitude > 0:
            self.use_energy(DRONE_HOVER_COST + DRONE_ALTITUDE_COST * self.altitude)

    def make_decision(self):
        """Rule-based decision model: the first matching rule gives the action."""
        if self.altitude == 0 and self.at_pad:
            if self.battery < DRONE_BATTERY:
                return "charge"
            return "ascend"                       # take off
        if self.state == RETURNING or self.should_return():
            return "return_to_pad"
        if self.altitude < CRUISE_ALTITUDE:
            return "ascend"
        if self.state == WAITING:
            return "wait_for_picker"
        if self.state == GOING_TO_CROP:
            return "fly_to_crop"
        if self.model.mode == BASIC and self.choose_crop_to_guard():
            return "fly_to_crop"
        return "explore"

    # ------------------------------------------------------------------ #
    # Perception
    # ------------------------------------------------------------------ #
    def scan(self):
        """Take pictures: ripe trees within a radius equal to the altitude."""
        if self.altitude < 1:
            return []
        found = self.model.ripe_trees_near(self.pos, self.altitude)
        new = 0
        for tree in found:
            if tree.seen_at is None:
                tree.seen_at = self.model.schedule.steps
                self.model.record_detection(tree)
                new += 1
        self.learn_sector_value(new)
        return found

    def should_return(self):
        if self.altitude == 0:
            return False
        pad = min(self.model.farm.drone_pad, key=lambda p: manhattan(self.pos, p))
        steps = manhattan(self.pos, pad) + self.altitude
        per_step = DRONE_MOVE_COST + DRONE_HOVER_COST + DRONE_ALTITUDE_COST * self.altitude
        return self.battery <= steps * per_step * 1.2 + DRONE_RESERVE

    def picker_arrived(self, tree_pos):
        return any(
            p.is_active and chebyshev(p.pos, tree_pos) <= ARM_REACH for p in self.model.pickers
        )

    # ------------------------------------------------------------------ #
    # Communication
    # ------------------------------------------------------------------ #
    def handle_messages(self):
        for m in self.model.bus.receive(self.unique_id):
            if m.performative == msg.PROPOSE:
                cfp = self.pending_cfps.get(m.content["task_id"])
                if cfp is None:
                    continue
                if "bundles" in m.content:
                    for b in m.content["bundles"]:
                        cfp["proposals"].append({"picker": m.sender, **b, "areas": tuple(b["areas"])})
                else:
                    cfp["proposals"].append({
                        "picker": m.sender,
                        "bid": m.content["cost"],
                        "cost": m.content.get("true_cost", m.content["cost"]),
                    })
            elif m.performative == msg.INFORM_SECTOR and self.player is not None:
                self.player.observe_choice(m.content["sector"])
            elif m.performative == msg.INFORM_DONE:
                self.model.contracts_completed += 1

    def new_patches(self, trees):
        """Group ripe trees by crop area, keeping only areas not announced recently."""
        now = self.model.schedule.steps
        by_patch = {}
        for t in trees:
            by_patch.setdefault(t.patch_id, []).append(t)
        fresh = {}
        for patch_id, patch_trees in by_patch.items():
            last = self.model.report_log.get(patch_id)
            if last is None or now - last >= REREPORT_INTERVAL:
                fresh[patch_id] = patch_trees
        return fresh

    def task_content(self, patch_id, trees):
        trees = sorted(trees, key=lambda t: -t.urgency)
        return {
            "patch": patch_id,
            "trees": [t.pos for t in trees],
            "urgency": round(trees[0].urgency, 3),
        }

    def report_crops(self, trees):
        """Extended mode: tell pickers where ripe crops are."""
        now = self.model.schedule.steps
        for patch_id, patch_trees in self.new_patches(trees).items():
            content = self.task_content(patch_id, patch_trees)
            if self.model.comm_mode == BROADCAST:
                receivers = [p.unique_id for p in self.model.pickers if p.is_active]
                if not receivers:
                    continue
                self.model.bus.broadcast(msg.INFORM_CROP, self.unique_id, receivers, content)
            else:
                picker = self.choose_picker(content["trees"][0])
                if picker is None:
                    continue                    # nobody free: try again later
                self.model.bus.send(msg.INFORM_CROP, self.unique_id, picker.unique_id, content)
            self.model.report_log[patch_id] = now
            self.reports += 1

    def choose_picker(self, tree_pos):
        """Direct messaging: the nearest active picker that still has free storage."""
        candidates = [p for p in self.model.pickers if p.is_active and p.spare_capacity > 0]
        if not candidates:
            return None
        return min(candidates, key=lambda p: manhattan(p.pos, tree_pos) + 10 * len(p.tasks))

    def call_for_proposals(self, trees):
        """Novel mode (Contract Net): announce each new crop area as a task."""
        now = self.model.schedule.steps
        receivers = [p.unique_id for p in self.model.pickers if p.is_active]
        if not receivers:
            return
        for patch_id, patch_trees in self.new_patches(trees).items():
            content = self.task_content(patch_id, patch_trees)
            content["task_id"] = self.model.new_task_id()
            self.model.bus.broadcast(msg.CFP, self.unique_id, receivers, content)
            self.pending_cfps[content["task_id"]] = {
                "content": content,
                "deadline": now + CFP_DEADLINE,
                "proposals": [],
            }
            self.model.report_log[patch_id] = now
            self.reports += 1

    def allocate(self, trees):
        """Novel mode: hand new crop areas to pickers with the model's allocation method."""
        allocation = self.model.allocation
        if allocation == CENTRAL_VRP:
            self.report_to_planner(trees)
        elif allocation == ENGLISH:
            self.run_english_auctions(trees)
        elif allocation in (COMBINATORIAL, LLM):
            now = self.model.schedule.steps
            for patch_id, patch_trees in self.new_patches(trees).items():
                self.pool[patch_id] = (patch_trees, now)
                self.model.report_log[patch_id] = now
        else:
            self.call_for_proposals(trees)

    def retry_later(self, patch_id):
        """Nobody took the area: allow a new announcement 10 steps from now."""
        self.model.report_log[patch_id] = self.model.schedule.steps - REREPORT_INTERVAL + 10

    def award_contracts(self):
        """After the deadline, award each task to the lowest bid and reject the rest."""
        now = self.model.schedule.steps
        for task_id, cfp in list(self.pending_cfps.items()):
            if now < cfp["deadline"]:
                continue
            del self.pending_cfps[task_id]
            if "areas" in cfp["content"]:
                self.award_bundles(task_id, cfp)
                continue
            if not cfp["proposals"]:
                self.retry_later(cfp["content"]["patch"])
                continue
            props = cfp["proposals"]
            win, payment = second_price(props) if self.model.allocation == VICKREY else first_price(props)
            self.model.bus.send(msg.ACCEPT, self.unique_id, win["picker"], cfp["content"])
            for p in props:
                if p["picker"] != win["picker"]:
                    self.model.bus.send(msg.REJECT, self.unique_id, p["picker"], {"task_id": task_id})
            self.model.record_award(win["picker"], payment, win["cost"], regret(win["cost"], props))

    # ---- English auction (descending clock) --------------------------- #
    def run_english_auctions(self, trees):
        """
        One descending-clock auction per new area. Clock rounds are radio exchanges,
        much faster than the robots move, so the whole auction runs within the step.
        Every round costs one price announcement plus one reply per remaining bidder.
        """
        bidders = [p for p in self.model.pickers if p.is_active]
        if not bidders:
            return
        now = self.model.schedule.steps
        for patch_id, patch_trees in self.new_patches(trees).items():
            content = self.task_content(patch_id, patch_trees)
            content["task_id"] = self.model.new_task_id()
            self.model.report_log[patch_id] = now
            self.reports += 1
            costs = {p.unique_id: p.estimate_cost(content) for p in bidders}
            winner, price, rounds, replies = english_clock(
                costs, ENGLISH_START_PRICE, ENGLISH_DECREMENT, self.random
            )
            self.model.bus.count(msg.CLOCK, rounds)
            self.model.bus.count(msg.PROPOSE, replies)
            self.model.english_rounds += rounds
            if winner is None:
                self.retry_later(patch_id)
                continue
            self.model.bus.send(msg.ACCEPT, self.unique_id, winner, content)
            valid = [{"picker": p, "bid": c, "cost": c} for p, c in costs.items() if c is not None]
            self.model.record_award(winner, price, costs[winner], regret(costs[winner], valid))

    # ---- areas gathered over a few steps (combinatorial auction, LLM) ---- #
    def flush_pool(self):
        """Offer the gathered areas once the oldest has waited BUNDLE_WINDOW steps."""
        now = self.model.schedule.steps
        for patch_id in [p for p, (t, _) in self.pool.items() if not any(x.is_ripe for x in t)]:
            del self.pool[patch_id]
        if not self.pool:
            return
        oldest = min(since for _, since in self.pool.values())
        if now - oldest < BUNDLE_WINDOW and len(self.pool) < MAX_BUNDLE_AREAS:
            return
        chosen = sorted(self.pool, key=lambda p: self.pool[p][1])[:MAX_BUNDLE_AREAS]
        areas = []
        for patch_id in chosen:
            ripe = [t for t in self.pool.pop(patch_id)[0] if t.is_ripe]
            areas.append(self.task_content(patch_id, ripe))
        if self.model.allocation == COMBINATORIAL:
            self.call_for_bundle_proposals(areas)
        else:
            self.ask_llm(areas)

    def call_for_bundle_proposals(self, areas):
        receivers = [p.unique_id for p in self.model.pickers if p.is_active]
        if not receivers:
            for a in areas:
                self.retry_later(a["patch"])
            return
        content = {"task_id": self.model.new_task_id(), "areas": areas}
        self.model.bus.broadcast(msg.CFP, self.unique_id, receivers, content)
        self.pending_cfps[content["task_id"]] = {
            "content": content,
            "deadline": self.model.schedule.steps + CFP_DEADLINE,
            "proposals": [],
        }
        self.reports += 1

    def award_bundles(self, task_id, cfp):
        """Combinatorial auction: solve winner determination, then award each bundle."""
        areas = {a["patch"]: a for a in cfp["content"]["areas"]}
        winners = solve_winner_determination(cfp["proposals"], AREA_VALUE)
        self.model.wdp_solves += 1
        sold = set()
        for w in winners:
            trees = [pos for a in w["areas"] for pos in areas[a]["trees"]]
            content = {
                "task_id": self.model.new_task_id(),
                "trees": trees,
                "urgency": max(areas[a]["urgency"] for a in w["areas"]),
                "patch": w["areas"][0],
            }
            self.model.bus.send(msg.ACCEPT, self.unique_id, w["picker"], content)
            self.model.record_award(w["picker"], w["bid"], w["cost"], None)
            sold |= set(w["areas"])
        winners_ids = {w["picker"] for w in winners}
        for picker in {p["picker"] for p in cfp["proposals"]} - winners_ids:
            self.model.bus.send(msg.REJECT, self.unique_id, picker, {"task_id": task_id})
        for patch_id in set(areas) - sold:
            self.retry_later(patch_id)

    def ask_llm(self, areas):
        """LLM coordinator: one prompt allocates every gathered area."""
        farm = self.model.farm
        pickers = [p for p in self.model.pickers
                   if p.is_active and not p.needs_recharge and p.spare_capacity > 0]
        if not pickers:
            for a in areas:
                self.retry_later(a["patch"])
            return
        info = []
        for p in pickers:
            start = list(p.tasks)[-1] if p.tasks else p.pos
            if start not in farm.passable:
                start = min(farm.pick_positions(start), key=lambda c: farm.base_distance.get(c, 1e9))
            dist = farm.distance_map([start])
            travel = {}
            for a in areas:
                reach = [dist[c] for c in farm.pick_positions(tuple(a["trees"][0])) if c in dist]
                travel[a["patch"]] = min(reach) if reach else 999
            info.append({"picker": p.unique_id, "battery": p.battery, "free_storage": p.spare_capacity,
                         "queued": len(p.tasks), "travel": travel})
        llm_areas = [{"area": a["patch"], "trees": len(a["trees"]), "urgency": a["urgency"]} for a in areas]
        plan = self.model.llm.assign(llm_areas, info)
        if plan is None:                       # LLM unavailable: nearest free picker instead
            self.model.llm_fallbacks += 1
            plan = {}
            for a in areas:
                chosen = self.choose_picker(a["trees"][0])
                if chosen is not None:
                    plan[a["patch"]] = chosen.unique_id
        by_patch = {a["patch"]: a for a in areas}
        for patch_id, picker_id in plan.items():
            content = dict(by_patch[patch_id], task_id=self.model.new_task_id())
            self.model.bus.send(msg.ACCEPT, self.unique_id, picker_id, content)
            self.model.record_award(picker_id, None, None, None)
        for patch_id in set(by_patch) - set(plan):
            self.retry_later(patch_id)

    def report_to_planner(self, trees):
        """Central planner: tell the farm manager about new areas (one message each)."""
        now = self.model.schedule.steps
        for patch_id, patch_trees in self.new_patches(trees).items():
            content = self.task_content(patch_id, patch_trees)
            self.model.bus.send(msg.INFORM_CROP, self.unique_id, self.model.planner.unique_id, content)
            self.model.report_log[patch_id] = now
            self.reports += 1

    # ---- the sector game ---------------------------------------------- #
    def learn_sector_value(self, new_crops):
        """Update how productive the current sector looks (new crops per scan)."""
        sectors = self.model.sectors
        if not sectors:
            return
        s = sector_of(sectors, self.pos)
        values = self.model.sector_values
        values[s] = (1 - SECTOR_LEARNING_RATE) * values[s] + SECTOR_LEARNING_RATE * new_crops
        if self.player is not None:
            self.player.observe_value(s, new_crops, SECTOR_LEARNING_RATE)

    def choose_sector(self):
        model = self.model
        active = sorted(d.unique_id for d in model.drones if d.is_active)
        if model.drone_strategy == NASH:
            choices = nash_assignment(model.sector_values, len(active))
            self.sector = choices[active.index(self.unique_id)]
        else:
            self.sector = self.player.choose(len(active) - 1)
            others = [d for d in active if d != self.unique_id]
            if others:
                model.bus.broadcast(msg.INFORM_SECTOR, self.unique_id, others, {"sector": self.sector})
        self.sector_since = model.schedule.steps
        model.sector_choices.append(self.sector)

    # ------------------------------------------------------------------ #
    # Basic mode: guarding a crop
    # ------------------------------------------------------------------ #
    def choose_crop_to_guard(self):
        """Pick a ripe tree in view whose area no other drone is already guarding."""
        for tree in sorted(self.findings, key=lambda t: manhattan(self.pos, t.pos)):
            if tree.patch_id not in self.model.guarded_patches:
                self.model.guarded_patches[tree.patch_id] = self.unique_id
                self.target = tree.pos
                self.state = GOING_TO_CROP
                return True
        return False

    def release_guard(self):
        if self.target is not None:
            patch_id = self.model.tree_at(self.target).patch_id
            if self.model.guarded_patches.get(patch_id) == self.unique_id:
                del self.model.guarded_patches[patch_id]
        self.target = None
        self.wait_steps = 0

    def fly_to_crop(self):
        if self.target is None or not self.model.tree_at(self.target).is_ripe:
            self.release_guard()
            self.state = EXPLORING
            return
        if self.pos == self.target:
            self.state = WAITING
            self.wait_steps = 0
            return
        self.fly_towards(self.target)

    def wait_for_picker(self):
        """Hover over the crop until a picker reaches it, then carry on exploring."""
        self.wait_steps += 1
        tree = self.model.tree_at(self.target)
        if self.picker_arrived(self.target) or not tree.is_ripe or self.wait_steps >= DRONE_MAX_WAIT:
            self.release_guard()
            self.state = EXPLORING

    # ------------------------------------------------------------------ #
    # Actions: the six movement directions, plus charging
    # ------------------------------------------------------------------ #
    def move_front(self):
        return self.fly(FRONT)

    def move_back(self):
        return self.fly(BACK)

    def move_left(self):
        return self.fly(LEFT)

    def move_right(self):
        return self.fly(RIGHT)

    def ascend(self):
        if self.altitude < MAX_ALTITUDE:
            self.altitude += 1
            self.use_energy(DRONE_MOVE_COST)
        if self.state == LANDED:
            self.state = EXPLORING

    def descend(self):
        over_tree = self.pos in self.model.farm.tree_cells
        floor = TREE_HEIGHT + 1 if over_tree else (0 if self.at_pad else 1)
        if self.altitude > floor:
            self.altitude -= 1
            self.use_energy(DRONE_MOVE_COST)

    def charge(self):
        self.state = LANDED
        self.battery = min(DRONE_BATTERY, self.battery + DRONE_CHARGE_RATE)

    def explore(self):
        """
        Random flight with momentum, sometimes changing altitude (2 <-> 3). With a
        sector strategy the drone first flies to its sector and then roams inside it.
        """
        self.state = EXPLORING
        bounds = (0, self.model.farm.width - 1, 0, self.model.farm.height - 1)
        if self.model.drone_strategy != ROAM and self.model.sectors:
            now = self.model.schedule.steps
            if self.sector is None or now - self.sector_since >= SECTOR_PERIOD:
                self.choose_sector()
            bounds = self.model.sectors[self.sector]

        def allowed(c):
            return bounds[0] <= c[0] <= bounds[1] and bounds[2] <= c[1] <= bounds[3]

        if self.sector is not None and self.model.drone_strategy != ROAM:
            if not allowed(self.pos):
                return self.fly_towards(sector_centre(self.model.sectors[self.sector]))
        if self.random.random() < ALTITUDE_CHANGE_PROB:
            if self.altitude < MAX_ALTITUDE:
                return self.ascend()
            return self.descend()
        ahead = (self.pos[0] + self.heading[0], self.pos[1] + self.heading[1])
        if not allowed(ahead) or self.random.random() < 0.1:
            reverse = (-self.heading[0], -self.heading[1])
            options = [
                d for d in HORIZONTAL
                if d != reverse and allowed((self.pos[0] + d[0], self.pos[1] + d[1]))
            ]
            self.heading = self.random.choice(options or [reverse])
        self.fly(self.heading)

    def return_to_pad(self):
        """Fly home, land and charge."""
        if self.state != RETURNING:
            self.release_guard()
            self.state = RETURNING
        pad = min(self.model.farm.drone_pad, key=lambda p: manhattan(self.pos, p))
        if self.pos != pad:
            self.fly_towards(pad)
        elif self.altitude > 0:
            self.descend()
        if self.altitude == 0:
            self.state = LANDED

    # ------------------------------------------------------------------ #
    # Movement helpers
    # ------------------------------------------------------------------ #
    def fly_towards(self, dest):
        dx, dy = dest[0] - self.pos[0], dest[1] - self.pos[1]
        if abs(dx) >= abs(dy):
            self.fly(RIGHT if dx > 0 else LEFT)
        else:
            self.fly(FRONT if dy > 0 else BACK)

    def fly(self, direction):
        """Move one cell horizontally. Climbs first if a tree is in the way."""
        nxt = (self.pos[0] + direction[0], self.pos[1] + direction[1])
        if not self.model.farm.in_bounds(nxt):
            return False
        if nxt in self.model.farm.tree_cells and self.altitude <= TREE_HEIGHT:
            self.ascend()
            return False
        self.model.grid.move_agent(self, nxt)
        self.distance += 1
        self.use_energy(DRONE_MOVE_COST)
        return True

    def use_energy(self, amount):
        self.battery -= amount
        self.energy_used += amount
        if self.battery <= 0:
            self.battery = 0
            self.state = OUT_OF_BATTERY
            self.release_guard()
