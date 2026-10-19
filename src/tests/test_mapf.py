"""Windowed Conflict-Based Search: low level, conflicts, CBS and the model."""
import unittest

from farm.constants import CBS, DIRECT, EXTENDED, INDEPENDENT
from farm.environment import FarmMap
from farm.mapf import cbs, first_conflict, prioritized, space_time_astar
from farm.model import FarmModel, blocked_steps


def corridor(width=7, water=()):
    """A 1-cell-wide corridor (row y=1) with walls (trees) above and below, plus a bay at x=3."""
    trees = [(x, y) for x in range(width) for y in (0, 2) if (x, y) != (3, 2)] + [(3, 3)]
    return FarmMap(width, 4, trees, water, [(0, 3)], [(width - 1, 3)])


def agent(farm, start, goal, wp=0):
    goals = frozenset([goal])
    return {"start": start, "wp": wp, "goals": goals, "h": farm.distance_map(list(goals))}


class TestLowLevel(unittest.TestCase):
    def test_shortest_path_without_constraints(self):
        farm = corridor()
        a = agent(farm, (0, 1), (6, 1))
        path, cost = space_time_astar(farm, a["start"], 0, a["goals"], a["h"], set(), set(), frozenset(), 12)
        self.assertEqual(path[0], (0, 1))
        self.assertEqual(path[-1], (6, 1))
        self.assertEqual(cost, 6)

    def test_vertex_constraint_forces_a_wait(self):
        farm = corridor()
        a = agent(farm, (0, 1), (6, 1))
        path, _ = space_time_astar(farm, a["start"], 0, a["goals"], a["h"], {((2, 1), 2)}, set(),
                                   frozenset(), 12)
        self.assertNotEqual(path[2], (2, 1))
        self.assertEqual(len(path) - 1, 7)              # one extra step

    def test_water_costs_an_extra_step(self):
        farm = corridor(water=[(2, 1)])
        a = agent(farm, (0, 1), (6, 1))
        path, cost = space_time_astar(farm, a["start"], 0, a["goals"], a["h"], set(), set(), frozenset(), 12)
        self.assertEqual(path.count((2, 1)), 2)         # waits once in the river cell
        self.assertEqual(cost, 7)

    def test_window_cuts_long_paths(self):
        farm = corridor(width=20)
        a = agent(farm, (0, 1), (19, 1))
        path, _ = space_time_astar(farm, a["start"], 0, a["goals"], a["h"], set(), set(), frozenset(), 5)
        self.assertEqual(len(path), 6)                  # t = 0..5


class TestConflicts(unittest.TestCase):
    def test_vertex_conflict(self):
        paths = {1: [(0, 1), (1, 1)], 2: [(2, 1), (1, 1)]}
        branches = first_conflict(paths, 5, set())
        self.assertEqual(branches[0], (1, "v", ((1, 1), 1)))

    def test_swap_conflict(self):
        paths = {1: [(0, 1), (1, 1)], 2: [(1, 1), (0, 1)]}
        kinds = {b[1] for b in first_conflict(paths, 5, set())}
        self.assertEqual(kinds, {"e"})

    def test_free_cells_never_conflict(self):
        paths = {1: [(0, 1), (1, 1)], 2: [(2, 1), (1, 1)]}
        self.assertIsNone(first_conflict(paths, 5, {(1, 1)}))


class TestCBS(unittest.TestCase):
    def test_head_on_robots_use_the_bay(self):
        farm = corridor()
        agents = {1: agent(farm, (0, 1), (6, 1)), 2: agent(farm, (6, 1), (0, 1))}
        paths, nodes, conflicts = cbs(agents, farm, 14, 500, set())
        self.assertIsNotNone(paths)
        self.assertIsNone(first_conflict(paths, 14, set()))
        self.assertGreater(conflicts, 0)
        self.assertTrue(any((3, 2) in p for p in paths.values()))   # someone pulled into the bay

    def test_prioritized_planning_is_conflict_free(self):
        farm = corridor()
        agents = {1: agent(farm, (0, 1), (6, 1)), 2: agent(farm, (6, 3), (0, 3))}
        paths = prioritized(agents, farm, 14, set())
        self.assertEqual(len(paths), 2)
        self.assertIsNone(first_conflict(paths, 14, set()))

    def test_prioritized_planning_is_incomplete(self):
        # Robot 1 plans first and ignores robot 2, which then cannot get out of the way
        # in time. CBS (above) solves the same instance: that is why CBS is tried first.
        farm = corridor()
        agents = {1: agent(farm, (0, 1), (6, 1)), 2: agent(farm, (6, 1), (0, 1))}
        paths = prioritized(agents, farm, 14, set())
        self.assertLess(len(paths), 2)
        self.assertIsNone(first_conflict(paths, 14, set()))


class TestCBSInModel(unittest.TestCase):
    def test_cbs_reduces_traffic_jams(self):
        def jams(planner):
            m = FarmModel(mode=EXTENDED, comm_mode=DIRECT, n_pickers=8, seed=3, max_steps=400,
                          path_planner=planner)
            while m.running:
                m.step()
            return m

        indep, joint = jams(INDEPENDENT), jams(CBS)
        self.assertLess(blocked_steps(joint), blocked_steps(indep))
        self.assertGreater(joint.paths.calls, 0)
        self.assertTrue(all(p.is_active for p in joint.pickers))


if __name__ == "__main__":
    unittest.main()
