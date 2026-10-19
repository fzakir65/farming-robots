"""Central planner: a farm manager that routes all pickers with OR-Tools.

This is the *centralised benchmark* for the decentralised methods. Drones report each
ripe area they find to the manager (one ``inform-crop`` message each). Every
``VRP_INTERVAL`` steps, or as soon as new areas arrive, the manager solves a
**capacitated vehicle-routing problem** (CVRP):

* vehicles: active pickers with free storage that are not heading home to recharge;
* each vehicle starts where its picker is and ends at the base;
* each crop area is a stop. Its demand is its number of ripe trees, and a vehicle's
  capacity is its free storage;
* an area may be left out, but at a penalty that grows with its urgency, so the
  solver prefers to visit crops that are about to rot.

OR-Tools' routing solver finds a good (not provably optimal) solution within the
time limit, using guided local search. Each picker is sent its stops in route
order.
"""
import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from . import messages as msg
from .constants import VRP_INTERVAL, VRP_TIME_LIMIT

UNREACHABLE = 10_000


class CentralPlanner:
    """Not a robot: a base-station computer with a radio link on the message bus."""

    def __init__(self, model, unique_id):
        self.model = model
        self.unique_id = unique_id
        model.bus.register(unique_id)
        self.known = {}             # patch id -> {"trees": [...], "urgency": u}
        self.assigned = {}          # patch id -> picker id
        self.last_plan = -VRP_INTERVAL
        self.new_reports = False
        self.solves = 0
        self.solve_seconds = 0.0

    def step(self):
        for m in self.model.bus.receive(self.unique_id):
            if m.performative == msg.INFORM_CROP:
                self.known[m.content["patch"]] = m.content
                self.new_reports = True
        now = self.model.schedule.steps
        if self.new_reports or now - self.last_plan >= VRP_INTERVAL:
            self.plan()

    # ------------------------------------------------------------------ #
    def open_areas(self):
        """Known areas that are still ripe and not being handled by a picker."""
        pickers = {p.unique_id: p for p in self.model.pickers}
        areas = []
        for patch, info in list(self.known.items()):
            ripe = [tuple(p) for p in info["trees"] if self.model.tree_at(tuple(p)).is_ripe]
            if not ripe:
                del self.known[patch]
                self.assigned.pop(patch, None)
                continue
            holder = pickers.get(self.assigned.get(patch))
            if holder is not None and holder.is_active and any(p in holder.tasks for p in ripe):
                continue
            urgency = max(self.model.tree_at(p).urgency for p in ripe)
            areas.append((patch, ripe, urgency))
        return areas

    def stop_cell(self, trees):
        """Where a picker parks to pick an area: the reachable cell nearest the base."""
        farm = self.model.farm
        cells = [c for t in trees for c in farm.pick_positions(t)]
        return min(cells, key=lambda c: (farm.base_distance.get(c, UNREACHABLE), c))

    def plan(self):
        self.last_plan = self.model.schedule.steps
        self.new_reports = False
        areas = self.open_areas()
        vehicles = [
            p for p in self.model.pickers
            if p.is_active and not p.needs_recharge and p.spare_capacity > 0
        ]
        if not areas or not vehicles:
            return
        t0 = time.perf_counter()
        routes = self.solve(areas, vehicles)
        self.solve_seconds += time.perf_counter() - t0
        self.solves += 1
        for picker, stops in routes:
            for order, (patch, ripe, urgency) in enumerate(stops):
                content = {
                    "patch": patch,
                    "trees": ripe,
                    "urgency": round(urgency, 3),
                    "task_id": self.model.new_task_id(),
                    "order": self.model.schedule.steps * 100 + order,
                }
                self.model.bus.send(msg.ACCEPT, self.unique_id, picker.unique_id, content)
                self.assigned[patch] = picker.unique_id
                self.model.contracts_awarded += 1

    def solve(self, areas, vehicles):
        farm = self.model.farm
        base = farm.picker_base[0]
        positions = [base] + [p.pos for p in vehicles] + [self.stop_cell(r) for _, r, _ in areas]
        n_v = len(vehicles)
        maps = [farm.distance_map([pos]) for pos in positions]
        dist = [[int(maps[i].get(positions[j], UNREACHABLE)) for j in range(len(positions))]
                for i in range(len(positions))]
        demand = [0] * (1 + n_v) + [len(r) for _, r, _ in areas]

        manager = pywrapcp.RoutingIndexManager(len(positions), n_v, list(range(1, n_v + 1)), [0] * n_v)
        routing = pywrapcp.RoutingModel(manager)

        def transit(i, j):
            return dist[manager.IndexToNode(i)][manager.IndexToNode(j)]

        routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(transit))

        def load(i):
            return demand[manager.IndexToNode(i)]

        routing.AddDimensionWithVehicleCapacity(
            routing.RegisterUnaryTransitCallback(load), 0,
            [max(0, v.spare_capacity) for v in vehicles], True, "Capacity",
        )
        for k, (_, ripe, urgency) in enumerate(areas):
            penalty = int((1000 + 2000 * urgency) * len(ripe))
            routing.AddDisjunction([manager.NodeToIndex(1 + n_v + k)], penalty)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromMilliseconds(int(VRP_TIME_LIMIT * 1000))
        solution = routing.SolveWithParameters(params)
        if solution is None:
            return []

        routes = []
        for v, picker in enumerate(vehicles):
            index = routing.Start(v)
            stops = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node > n_v:
                    stops.append(areas[node - 1 - n_v])
                index = solution.Value(routing.NextVar(index))
            if stops:
                routes.append((picker, stops))
        return routes
