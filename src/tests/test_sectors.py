"""The drones' sector game: equilibria, fictitious play and the strategies in the model."""
import unittest

import numpy as np

from farm.constants import DIRECT, EXTENDED, FICTITIOUS, NASH
from farm.model import FarmModel, mean_detect_latency
from farm.sectors import (FictitiousPlayer, best_response, is_nash, make_sectors, mixed_equilibria_2p,
                          nash_assignment, potential, sector_of)


class TestSectorGame(unittest.TestCase):
    def test_two_drones_split_when_second_sector_is_worth_it(self):
        self.assertEqual(sorted(nash_assignment([10, 6, 1, 1], 2)), [0, 1])

    def test_two_drones_share_a_dominant_sector(self):
        # 10/2 = 5 > 3, so both stay in sector 0
        self.assertEqual(nash_assignment([10, 3, 1, 1], 2), [0, 0])

    def test_result_is_a_nash_equilibrium(self):
        values = [4.0, 3.0, 2.0, 1.0]
        for n in range(1, 7):
            self.assertTrue(is_nash(values, nash_assignment(values, n)))

    def test_best_response_counts_congestion(self):
        self.assertEqual(best_response([10, 6], others=[0]), 1)
        self.assertEqual(best_response([10, 6], others=[1]), 0)

    def test_potential_rises_with_an_improving_move(self):
        values = [10, 6]
        self.assertGreater(potential(values, [0, 1]), potential(values, [0, 0]))

    def test_nashpy_agrees_with_best_response_dynamics(self):
        values = [10.0, 6.0, 1.0, 1.0]
        pure = nash_assignment(values, 2)
        equilibria = mixed_equilibria_2p(values)
        as_pure = [(int(np.argmax(p)), int(np.argmax(q))) for p, q in equilibria
                   if max(p) == 1.0 and max(q) == 1.0]
        self.assertIn(tuple(pure), as_pure)

    def test_fictitious_player_avoids_crowded_sector(self):
        player = FictitiousPlayer(2, [10.0, 6.0])
        for _ in range(20):
            player.observe_choice(0)          # the other drone always searches sector 0
        self.assertEqual(player.choose(n_others=1), 1)


class TestSectorsOnTheFarm(unittest.TestCase):
    def test_four_sectors_cover_the_farm(self):
        model = FarmModel(seed=1)
        sectors = make_sectors(model.farm)
        self.assertEqual(len(sectors), 4)
        for x in range(model.grid.width):
            for y in range(model.grid.height):
                s = sector_of(sectors, (x, y))
                x0, x1, y0, y1 = sectors[s]
                self.assertTrue(x0 <= x <= x1 and y0 <= y <= y1)

    def test_strategies_run_and_detect_crops(self):
        for strategy in (NASH, FICTITIOUS):
            with self.subTest(strategy=strategy):
                m = FarmModel(mode=EXTENDED, comm_mode=DIRECT, drone_strategy=strategy, seed=2, max_steps=400)
                while m.running:
                    m.step()
                self.assertGreater(len(m.sector_choices), 0)
                self.assertGreater(mean_detect_latency(m), 0)
                self.assertGreater(m.picked, 0)

    def test_drones_stay_in_their_sector_once_there(self):
        m = FarmModel(mode=EXTENDED, drone_strategy=NASH, seed=2, max_steps=300)
        inside = total = 0
        while m.running:
            m.step()
            for d in m.drones:
                if d.state == "exploring" and d.sector is not None:
                    x0, x1, y0, y1 = m.sectors[d.sector]
                    total += 1
                    inside += x0 <= d.pos[0] <= x1 and y0 <= d.pos[1] <= y1
        self.assertGreater(inside / total, 0.7)


if __name__ == "__main__":
    unittest.main()
