"""Multi-agent path finding: windowed Conflict-Based Search (CBS).

Independent planning (Dijkstra per robot) ignores other robots, so pickers bump into
each other in the narrow lanes and fall back on waiting, detours and squeezing past.
CBS (Sharon et al., 2015) plans all moving pickers *together*:

* the **low level** is a space-time A* for one robot that respects a set of
  constraints ("not in cell c at time t", "not along edge a->b at time t");
* the **high level** searches a tree of constraint sets. It finds the first
  conflict between the robots' paths and branches: in one child the first robot
  must avoid it, in the other the second robot must.

Two practical changes, as in lifelong multi-robot systems:

* **Windowed** (rolling horizon, as in WHCA*/RHCR): conflicts are only resolved
  ``window`` steps ahead. Beyond the window each path continues along its
  independent shortest route, and the plan is refreshed every half window or when a
  robot deviates.
* Robots act one after another within a step (random activation), so a robot cannot
  enter a cell another robot is still leaving. **Following conflicts** (entering a
  cell occupied in the previous step) are therefore resolved as well (1-robust
  plans).

Water: a robot that enters a river cell must spend ``WATER_SLOWDOWN - 1`` extra steps
there before it can leave, so the search state includes that progress. If the
high-level budget runs out, **prioritised planning** (one robot after another, each
avoiding the paths already reserved) is used instead.
"""
import heapq
import itertools

from .constants import WATER_SLOWDOWN

BIG = 10_000


def _at(path, t):
    return path[t] if t < len(path) else path[-1]


def space_time_astar(farm, start, start_wp, goals, h, vertex, edges, static, window,
                     max_expansions=6000):
    """
    Path for one robot as a list of cells for t = 0..T (T <= window), with waits as
    repeated cells. It ends at a goal cell (where the robot may then stay until the
    window ends) or, if the goal is further away, at time ``window``.
    Returns (path, cost) or (None, None).
    """
    last_wait = WATER_SLOWDOWN - 1
    counter = itertools.count()
    heap = [(h.get(start, BIG), next(counter), 0, start, start_wp)]
    parent = {(start, 0, start_wp): None}
    closed = set()
    expansions = 0
    while heap and expansions < max_expansions:
        f, _, t, cell, wp = heapq.heappop(heap)
        state = (cell, t, wp)
        if state in closed:
            continue
        closed.add(state)
        expansions += 1
        at_goal = cell in goals and not any((cell, tt) in vertex for tt in range(t + 1, window + 1))
        if at_goal or t == window:
            path = []
            while state is not None:
                path.append(state[0])
                state = parent[state]
            return path[::-1], t + (0 if at_goal else h.get(cell, BIG))
        in_water = farm.is_water(cell)
        moves = [(cell, min(wp + 1, last_wait) if in_water else 0)]          # wait
        if not in_water or wp >= last_wait:
            moves += [(n, 0) for n in farm.neighbours(cell)]
        nt = t + 1
        for nc, nwp in moves:
            if nc != start and nc in static:
                continue
            if (nc, nt) in vertex or ((cell, nc), nt) in edges:
                continue
            nstate = (nc, nt, nwp)
            if nstate in closed or nstate in parent:
                continue
            parent[nstate] = state
            heapq.heappush(heap, (nt + h.get(nc, BIG), next(counter), nt, nc, nwp))
    return None, None


def first_conflict(paths, window, free):
    """
    The earliest conflict between two paths, or None. Returns a list of
    ``(agent, kind, constraint)`` branches, where kind is "v" (vertex) or "e" (edge).
    Cells in ``free`` (the base station) never conflict.
    """
    ids = sorted(paths)
    for t in range(1, window + 1):
        for a, b in itertools.combinations(ids, 2):
            pa, pb = paths[a], paths[b]
            a0, a1, b0, b1 = _at(pa, t - 1), _at(pa, t), _at(pb, t - 1), _at(pb, t)
            if a1 == b1 and a1 not in free:
                return [(a, "v", (a1, t)), (b, "v", (b1, t))]
            if a1 == b0 and b1 == a0 and a0 != a1:
                return [(a, "e", ((a0, a1), t)), (b, "e", ((b0, b1), t))]
            # following: entering the cell the other robot occupied one step earlier
            for x, x0, x1, y, y0 in ((a, a0, a1, b, b0), (b, b0, b1, a, a0)):
                if x1 == y0 and x0 != x1 and x1 not in free:
                    branches = [(x, "v", (x1, t))]
                    if t - 1 > 0:
                        branches.append((y, "v", (y0, t - 1)))
                    return branches
    return None


def _plan(agent, farm, cons, static, window):
    vertex, edges = cons
    return space_time_astar(farm, agent["start"], agent["wp"], agent["goals"], agent["h"],
                            vertex, edges, static, window)


def cbs(agents, farm, window, max_nodes, free, static=frozenset()):
    """
    Conflict-Based Search. ``agents``: {id: {"start", "wp", "goals", "h"}}.
    Returns ({id: path}, nodes expanded, conflicts resolved) or (None, nodes, conflicts).
    """
    cons = {i: (frozenset(), frozenset()) for i in agents}
    paths, cost = {}, 0
    for i, ag in agents.items():
        p, c = _plan(ag, farm, cons[i], static, window)
        if p is None:
            return None, 0, 0
        paths[i], cost = p, cost + c
    counter = itertools.count()
    heap = [(cost, next(counter), cons, paths)]
    nodes = conflicts = 0
    while heap and nodes < max_nodes:
        cost, _, cons, paths = heapq.heappop(heap)
        nodes += 1
        conflict = first_conflict(paths, window, free)
        if conflict is None:
            return paths, nodes, conflicts
        conflicts += 1
        for agent, kind, item in conflict:
            vertex, edges = cons[agent]
            new = (vertex | {item}, edges) if kind == "v" else (vertex, edges | {item})
            p, c = _plan(agents[agent], farm, new, static, window)
            if p is None:
                continue
            child_cons = dict(cons)
            child_cons[agent] = new
            child_paths = dict(paths)
            child_paths[agent] = p
            child_cost = sum(len(q) for q in child_paths.values())
            heapq.heappush(heap, (child_cost, next(counter), child_cons, child_paths))
    return None, nodes, conflicts


def prioritized(agents, farm, window, free, static=frozenset()):
    """Plan robots one at a time; each avoids the space-time cells already reserved."""
    reserved, swaps, paths = set(), set(), {}
    for i in sorted(agents):
        vertex = set(reserved)
        p, _ = space_time_astar(farm, agents[i]["start"], agents[i]["wp"], agents[i]["goals"],
                                agents[i]["h"], vertex, swaps, static, window)
        if p is None:
            continue
        paths[i] = p
        for t in range(window + 1):
            cell = _at(p, t)
            if cell in free:
                continue
            reserved.add((cell, t))
            reserved.add((cell, t + 1))          # nobody follows into it next step
            prev = _at(p, t - 1) if t > 0 else cell
            if prev != cell:
                swaps.add(((cell, prev), t))     # nobody swaps with this robot
    return paths


class PathCoordinator:
    """Runs windowed CBS for every moving picker at the start of a step when needed."""

    def __init__(self, model, window, max_nodes):
        self.model = model
        self.window = window
        self.max_nodes = max_nodes
        self.requests = set()
        self.expected = {}          # picker id -> {time: planned cell}
        self.last_plan = -window
        self._h_cache = {}
        # statistics
        self.calls = 0
        self.conflicts = 0
        self.fallbacks = 0

    def request(self, picker):
        self.requests.add(picker.unique_id)

    def heuristic(self, goals):
        key = frozenset(goals)
        h = self._h_cache.get(key)
        if h is None:
            if len(self._h_cache) > 512:
                self._h_cache.clear()
            h = self._h_cache[key] = self.model.farm.distance_map(list(key))
        return h

    def before_step(self):
        model = self.model
        now = model.schedule.steps
        movers = [p for p in model.pickers if p.is_active and p.path and p.goals]
        if not movers:
            self.requests.clear()
            return
        deviated = any(
            now in self.expected.get(p.unique_id, {}) and self.expected[p.unique_id][now] != p.pos
            for p in movers
        )
        if not (self.requests or deviated or now - self.last_plan >= self.window // 2):
            return
        free = set(model.farm.picker_base)
        moving = {p.unique_id for p in movers}
        static = frozenset(p.pos for p in model.pickers
                           if p.is_active and p.unique_id not in moving and p.pos not in free)
        agents = {
            p.unique_id: {"start": p.pos, "wp": p.water_progress, "goals": frozenset(p.goals),
                          "h": self.heuristic(p.goals)}
            for p in movers
        }
        paths, _, conflicts = cbs(agents, model.farm, self.window, self.max_nodes, free, static)
        self.calls += 1
        self.conflicts += conflicts
        if paths is None:
            self.fallbacks += 1
            paths = prioritized(agents, model.farm, self.window, free, static)
        for p in movers:
            path = paths.get(p.unique_id)
            if path is None:
                continue
            tail = []
            if path[-1] not in p.goals:
                tail = model.farm.shortest_path(path[-1], p.goals) or []
            p.path = path[1:] + tail
            self.expected[p.unique_id] = {now + t: cell for t, cell in enumerate(path)}
        self.requests.clear()
        self.last_plan = now
