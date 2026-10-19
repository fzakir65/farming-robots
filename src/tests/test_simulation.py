"""End-to-end tests: every mode runs without errors and the KPIs are recorded."""
import unittest

from farm.constants import BASIC, BROADCAST, DIRECT, EXTENDED, MODES, NOVEL
from farm.model import FarmModel


def run(model, steps):
    for _ in range(steps):
        if not model.running:
            break
        model.step()
    return model


class TestSimulationRuns(unittest.TestCase):
    def test_every_mode_runs(self):
        for mode in MODES:
            for random_layout in (False, True):
                with self.subTest(mode=mode, random_layout=random_layout):
                    model = run(FarmModel(mode=mode, random_layout=random_layout, seed=7), 300)
                    self.assertGreater(model.picked, 0)
                    self.assertTrue(all(p.is_active for p in model.pickers))

    def test_both_extended_comm_modes_run(self):
        for comm in (BROADCAST, DIRECT):
            with self.subTest(comm=comm):
                model = run(FarmModel(mode=EXTENDED, comm_mode=comm, seed=3), 300)
                self.assertGreater(model.bus.sent, 0)

    def test_basic_mission_completes(self):
        model = run(FarmModel(mode=BASIC, seed=5), 2000)
        self.assertFalse(model.running)
        self.assertEqual(model.delivered, model.picked)
        self.assertEqual(sum(1 for t in model.trees if t.is_ripe), 0)

    def test_extended_mode_regrows_crops(self):
        model = run(FarmModel(mode=EXTENDED, seed=5), 600)
        self.assertGreater(model.replanted, 0)

    def test_novel_mode_completes_contracts(self):
        model = run(FarmModel(mode=NOVEL, seed=5), 400)
        self.assertGreater(model.contracts_awarded, 0)
        self.assertGreater(model.contracts_completed, 0)

    def test_stops_at_max_steps(self):
        model = run(FarmModel(mode=EXTENDED, max_steps=50, seed=1), 100)
        self.assertEqual(model.schedule.steps, 50)
        self.assertFalse(model.running)

    def test_same_seed_same_result(self):
        a = run(FarmModel(mode=NOVEL, seed=11), 200)
        b = run(FarmModel(mode=NOVEL, seed=11), 200)
        self.assertEqual((a.picked, a.delivered, a.bus.sent), (b.picked, b.delivered, b.bus.sent))

    def test_kpis_are_collected_every_step(self):
        model = run(FarmModel(mode=EXTENDED, seed=2), 30)
        df = model.datacollector.get_model_vars_dataframe()
        self.assertEqual(len(df), 31)          # initial state + 30 steps
        for kpi in ("Delivered", "Wasted", "Harvest rate", "Energy per strawberry", "Mean pick latency"):
            self.assertIn(kpi, df.columns)


if __name__ == "__main__":
    unittest.main()
