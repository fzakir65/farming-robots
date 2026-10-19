"""The farm environment: static agents (trees, river, base stations, lanes),
layout generation and path finding.

Static agents are placed on the grid so they can be drawn and sensed, but they are
not added to the scheduler. Crop ageing is driven by the model instead (see
``FarmModel.grow_crops``), because the environment, not the robots, handles it.
"""
import heapq

from .compat import FarmAgent
from .constants import (
    ARM_REACH,
    LANE_WIDTH,
    PATCH_LENGTH,
    RIPE_AGE,
    RIVER_WIDTH,
    ROT_AGE,
    TREE_ROW_WIDTH,
    WATER_SLOWDOWN,
)

# Crop states of a tree
GROWING = "growing"
RIPE = "ripe"
EMPTY = "empty"

# Kinds of base station
PICKER_BASE = "picker_base"
DRONE_PAD = "drone_pad"

# 4-neighbourhood used by ground robots (front, back, left, right)
STEPS_4 = [(0, 1), (0, -1), (-1, 0), (1, 0)]


class Ground(FarmAgent):
    """A walkable lane cell (brown soil)."""


class Water(FarmAgent):
    """A river cell. Pickers can cross it, but slowly."""


class Base(FarmAgent):
    """A base station cell where robots unload and recharge."""

    def __init__(self, unique_id, model, kind):
        super().__init__(unique_id, model)
        self.kind = kind


class Tree(FarmAgent):
    """A strawberry tree. Its crop ages, ripens, rots and is replanted."""

    def __init__(self, unique_id, model, patch_id, state=GROWING, age=0):
        super().__init__(unique_id, model)
        self.patch_id = patch_id
        self.state = state
        self.age = age
        self.ripe_since = 0 if state == RIPE else None
        self.seen_at = None     # step a drone first saw it ripe (detection latency KPI)

    @property
    def is_ripe(self):
        return self.state == RIPE

    @property
    def urgency(self):
        """
        How urgently the crop must be collected, in [0, 1].

        This stands in for the drones' computer-vision estimate: 0 means just ripe,
        1 means about to rot. Without ageing (basic mode) every ripe crop is 0.5.
        """
        if not self.is_ripe:
            return 0.0
        if not self.model.crops_age:
            return 0.5
        return min(1.0, max(0.0, (self.age - RIPE_AGE) / (ROT_AGE - RIPE_AGE)))

    def grow(self):
        """Advance the crop life-cycle by one step: growing -> ripe -> (rotten) empty."""
        if self.state == EMPTY:
            return
        self.age += 1
        if self.state == GROWING and self.age >= RIPE_AGE:
            self.ripen()
        elif self.state == RIPE and self.age >= ROT_AGE:
            self.state = EMPTY
            self.model.wasted += 1

    def ripen(self):
        self.state = RIPE
        self.age = max(self.age, RIPE_AGE)
        self.ripe_since = self.model.schedule.steps
        self.seen_at = None

    def harvest(self):
        """Cut the strawberries. Returns True if there was something to pick."""
        if not self.is_ripe:
            return False
        self.state = EMPTY
        self.model.record_pick(self)
        return True

    def replant(self, delay=0, bears_fruit=True):
        """Seed new strawberries, restarting the ageing cycle after ``delay`` steps.

        A tree that does not bear fruit this cycle stays empty until its area is re-seeded.
        """
        self.state = GROWING if bears_fruit else EMPTY
        self.age = -delay
        self.ripe_since = None


def build_layout(width, height, rng, random_layout=False):
    """
    Work out where the river, tree rows and base stations go.

    Returns a dict with:
      * ``river``: set of river columns
      * ``strips``: list of (x0, y0, y1); each strip is TREE_ROW_WIDTH trees wide
      * ``picker_base``: list of cells for the picker base station (bottom-left)
      * ``drone_pad``: list of cells for the drone landing pad (top, left of the river)

    The default layout mirrors Figure 1 of the brief: rows of trees separated by
    lanes, with a river (and a walkable bank on each side) splitting the field in
    two. Every tree row has walkable cells on both sides, so a picker's arm can
    reach every tree. With ``random_layout`` the river position and the length of
    every row are randomised.
    """
    # edge lane + n rows of 3 + (n-2) lanes + bank/river/bank + edge lane must fit
    unit = TREE_ROW_WIDTH + LANE_WIDTH
    n_strips = max(2, (width + 2 * LANE_WIDTH - RIVER_WIDTH - 4) // unit)
    river_after = rng.randint(1, n_strips - 1) if random_layout else n_strips // 2

    strips = []
    river = set()
    x = 1                                   # column 0 is an edge lane
    for i in range(n_strips):
        if i == river_after:
            river = set(range(x + 1, x + 1 + RIVER_WIDTH))   # bank at x, then the river
            x += RIVER_WIDTH + 2                             # ... and the far bank
        elif i > 0:
            x += LANE_WIDTH
        if random_layout:
            y0 = rng.randint(2, 5)
            y1 = rng.randint(height - 6, height - 2)
        else:
            y0 = 2 + (i % 3) * 2
            y1 = height - 2 - (i % 2) * 3
        strips.append((x, y0, y1))
        x += TREE_ROW_WIDTH

    # Pickers live at the bottom-left; drones use a pad at the top of the first lane
    picker_base = [(0, 0), (1, 0), (2, 0), (3, 0)]
    first_lane = strips[0][0] + TREE_ROW_WIDTH
    drone_pad = [(first_lane, height - 1), (first_lane + 1, height - 1)]
    return {
        "river": river,
        "strips": strips,
        "picker_base": picker_base,
        "drone_pad": drone_pad,
    }


class FarmMap:
    """
    Static knowledge about the terrain, shared by all agents.

    It answers "can a picker stand here?", "how expensive is it to move?" and
    "what is the cheapest route?". Routes use Dijkstra's algorithm, where leaving a
    water cell costs WATER_SLOWDOWN steps instead of 1.
    """

    def __init__(self, width, height, tree_cells, water_cells, picker_base, drone_pad):
        self.width = width
        self.height = height
        self.tree_cells = set(tree_cells)
        self.water_cells = set(water_cells)
        self.picker_base = list(picker_base)
        self.drone_pad = list(drone_pad)
        self.passable = {
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) not in self.tree_cells
        }
        # distance (in steps) from every passable cell to the picker base
        self.base_distance = self.distance_map(self.picker_base)
        self._pick_positions = {}

    def in_bounds(self, pos):
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def is_passable(self, pos):
        return pos in self.passable

    def is_water(self, pos):
        return pos in self.water_cells

    def move_cost(self, from_pos):
        """Steps needed to leave ``from_pos`` (robots are slowed down in water)."""
        return WATER_SLOWDOWN if from_pos in self.water_cells else 1

    def neighbours(self, pos, blocked=frozenset()):
        for dx, dy in STEPS_4:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in self.passable and nxt not in blocked:
                yield nxt

    def distance_map(self, sources, blocked=frozenset()):
        """Dijkstra from several sources. Returns {cell: cost} for every reachable cell.

        Costs are symmetric enough for our use (water only ever doubles a step),
        so the map is also used as "distance *to* the sources".
        """
        dist = {}
        heap = [(0, s) for s in sources]
        heapq.heapify(heap)
        while heap:
            d, cell = heapq.heappop(heap)
            if cell in dist:
                continue
            dist[cell] = d
            step = self.move_cost(cell)
            for nxt in self.neighbours(cell, blocked):
                if nxt not in dist:
                    heapq.heappush(heap, (d + step, nxt))
        return dist

    def shortest_path(self, start, goals, blocked=frozenset()):
        """
        Cheapest route from ``start`` to any cell in ``goals``.

        Returns the list of cells to visit (``start`` excluded), [] if already at a
        goal, or None if no goal can be reached.
        """
        goals = set(goals)
        if start in goals:
            return []
        prev = {start: None}
        dist = {start: 0}
        heap = [(0, start)]
        while heap:
            d, cell = heapq.heappop(heap)
            if d > dist[cell]:
                continue
            if cell in goals:
                path = []
                while cell != start:
                    path.append(cell)
                    cell = prev[cell]
                return path[::-1]
            step = self.move_cost(cell)
            for nxt in self.neighbours(cell, blocked):
                nd = d + step
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    prev[nxt] = cell
                    heapq.heappush(heap, (nd, nxt))
        return None

    def pick_positions(self, tree_pos):
        """Passable cells from which a picker's arm can reach ``tree_pos``."""
        if tree_pos not in self._pick_positions:
            tx, ty = tree_pos
            self._pick_positions[tree_pos] = [
                (x, y)
                for x in range(tx - ARM_REACH, tx + ARM_REACH + 1)
                for y in range(ty - ARM_REACH, ty + ARM_REACH + 1)
                if (x, y) in self.passable
            ]
        return self._pick_positions[tree_pos]


def patches_for_strip(x0, y0, y1):
    """Split a row of trees into 3x3 crop areas. Yields lists of cells."""
    y = y0
    while y <= y1:
        top = min(y + PATCH_LENGTH - 1, y1)
        yield [(x, yy) for x in range(x0, x0 + TREE_ROW_WIDTH) for yy in range(y, top + 1)]
        y = top + 1
