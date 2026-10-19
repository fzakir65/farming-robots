"""Property-based tests (Hypothesis): invariants that must hold for *any* farm.

Instead of hand-picked examples, Hypothesis generates many random seeds, layouts,
fleet sizes and option combinations, and shrinks any failure to a minimal case.
"""
import unittest

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from farm.auctions import english_clock, first_price, second_price, solve_winner_determination
from farm.constants import (ALLOCATIONS, BASIC, CBS, DRONE_STRATEGIES, EXTENDED, INDEPENDENT, LLM,
                            MAX_ALTITUDE, MODES, NOVEL, PICKER_BATTERY, TREE_HEIGHT)
from farm.environment import RIPE
from farm.mapf import cbs, first_conflict
from farm.model import FarmModel, harvest_rate
from farm.sectors import is_nash, nash_assignment

FAST = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def run(model, steps):
    for _ in range(steps):
        if not model.running:
            break
        model.step()
    return model


class TestSimulationInvariants(unittest.TestCase):
    @FAST
    @given(
        seed=st.integers(0, 10_000),
        mode=st.sampled_from(MODES),
        allocation=st.sampled_from([a for a in ALLOCATIONS if a != LLM]),
        planner=st.sampled_from([INDEPENDENT, CBS]),
        drones=st.sampled_from(DRONE_STRATEGIES),
        n_pickers=st.integers(1, 7),
        capacity=st.integers(2, 15),
        random_layout=st.booleans(),
    )
    def test_conservation_and_bounds(self, seed, mode, allocation, planner, drones, n_pickers, capacity,
                                     random_layout):
        m = run(FarmModel(mode=mode, allocation=allocation, path_planner=planner, drone_strategy=drones,
                          n_pickers=n_pickers, picker_capacity=capacity, random_layout=random_layout,
                          seed=seed, max_steps=150), 150)
        # every strawberry picked is either delivered or still being carried
        self.assertEqual(m.picked, m.delivered + sum(p.load for p in m.pickers))
        for p in m.pickers:
            self.assertLessEqual(p.load, p.capacity)
            self.assertTrue(0 <= p.battery <= PICKER_BATTERY)
            self.assertIn(p.pos, m.farm.passable)                 # never inside a tree
            for pos in p.tasks:
                self.assertIn(pos, m.trees_by_pos)
        for d in m.drones:
            self.assertTrue(0 <= d.altitude <= MAX_ALTITUDE)
            if d.pos in m.farm.tree_cells:
                self.assertGreater(d.altitude, TREE_HEIGHT)       # never crashes into a tree
        self.assertTrue(0.0 <= harvest_rate(m) <= 1.0)
        if mode == BASIC:
            self.assertEqual(m.wasted, 0)                         # crops do not age in basic mode

    @FAST
    @given(seed=st.integers(0, 10_000), w=st.integers(24, 40), h=st.integers(20, 34))
    def test_every_tree_is_reachable(self, seed, w, h):
        m = FarmModel(width=w, height=h, random_layout=True, seed=seed)
        for pos in m.trees_by_pos:
            reach = [c for c in m.farm.pick_positions(pos) if c in m.farm.base_distance]
            self.assertTrue(reach, f"tree {pos} cannot be reached")

    @FAST
    @given(seed=st.integers(0, 10_000))
    def test_same_seed_same_run(self, seed):
        a = run(FarmModel(mode=NOVEL, seed=seed, max_steps=80), 80)
        b = run(FarmModel(mode=NOVEL, seed=seed, max_steps=80), 80)
        self.assertEqual((a.picked, a.bus.sent, a.wasted), (b.picked, b.bus.sent, b.wasted))

    @FAST
    @given(seed=st.integers(0, 10_000))
    def test_ripe_trees_have_a_ripening_time(self, seed):
        m = run(FarmModel(mode=EXTENDED, seed=seed, max_steps=120), 120)
        for t in m.trees:
            if t.state == RIPE:
                self.assertIsNotNone(t.ripe_since)


bids = st.lists(
    st.fixed_dictionaries({"picker": st.integers(1, 6), "bid": st.floats(1, 100), "cost": st.floats(1, 100)}),
    min_size=1, max_size=6,
)


class TestMechanismProperties(unittest.TestCase):
    @given(bids)
    def test_second_price_never_pays_less_than_first_price(self, proposals):
        w1, p1 = first_price(proposals)
        w2, p2 = second_price(proposals)
        self.assertEqual(w1["picker"], w2["picker"])
        self.assertGreaterEqual(p2, p1)

    @given(st.dictionaries(st.integers(1, 8), st.one_of(st.none(), st.floats(1, 110)), min_size=1, max_size=8),
           st.integers(0, 1000))
    def test_english_clock_picks_a_cheapest_bidder(self, costs, seed):
        import random

        winner, price, _, _ = english_clock(costs, 120.0, 4.0, random.Random(seed))
        valid = {p: c for p, c in costs.items() if c is not None}
        if winner is None:
            self.assertTrue(all(c > 120.0 for c in valid.values()))
        else:
            self.assertGreaterEqual(price, valid[winner])                  # never below its cost
            self.assertLessEqual(valid[winner], min(valid.values()) + 4.0)  # within one decrement

    @settings(max_examples=40, deadline=None)
    @given(st.lists(
        st.fixed_dictionaries({
            "picker": st.integers(1, 4),
            "areas": st.lists(st.sampled_from("abcde"), min_size=1, max_size=3, unique=True).map(tuple),
            "trees": st.integers(1, 9),
            "bid": st.floats(1, 200),
        }), min_size=1, max_size=12))
    def test_winner_determination_is_feasible_and_profitable(self, raw):
        for b in raw:
            b["cost"] = b["bid"]
        winners = solve_winner_determination(raw, value_per_tree=20.0)
        areas = [a for w in winners for a in w["areas"]]
        self.assertEqual(len(areas), len(set(areas)))                       # each area sold once
        pickers = [w["picker"] for w in winners]
        self.assertEqual(len(pickers), len(set(pickers)))                   # one bundle per picker
        surplus = sum(20.0 * w["trees"] - w["bid"] for w in winners)
        best_single = max([20.0 * b["trees"] - b["bid"] for b in raw] + [0.0])
        self.assertGreaterEqual(surplus + 1e-6, best_single)                # at least the best bid

    @given(st.lists(st.floats(0.1, 50), min_size=2, max_size=5), st.integers(1, 8))
    def test_best_response_dynamics_reaches_nash(self, values, n):
        self.assertTrue(is_nash(values, nash_assignment(values, n)))


class TestCBSProperty(unittest.TestCase):
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(seed=st.integers(0, 10_000), n=st.integers(2, 5))
    def test_cbs_paths_are_conflict_free(self, seed, n):
        import random

        m = FarmModel(seed=seed)
        rng = random.Random(seed)
        cells = sorted(c for c in m.farm.passable if c not in m.farm.water_cells and c[1] > 0)
        chosen = rng.sample(cells, 2 * n)
        agents = {}
        for i in range(n):
            goals = frozenset([chosen[n + i]])
            agents[i] = {"start": chosen[i], "wp": 0, "goals": goals, "h": m.farm.distance_map(list(goals))}
        paths, _, _ = cbs(agents, m.farm, 10, 200, set(m.farm.picker_base))
        if paths is not None:
            self.assertIsNone(first_conflict(paths, 10, set(m.farm.picker_base)))


if __name__ == "__main__":
    unittest.main()
