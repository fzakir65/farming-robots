"""Auction mechanisms, winner determination, and the auction modes end to end."""
import random
import unittest

from farm import messages as msg
from farm.auctions import english_clock, first_price, regret, second_price, solve_winner_determination
from farm.constants import COMBINATORIAL, CONTRACT_NET, ENGLISH, NOVEL, VICKREY
from farm.model import FarmModel, mean_regret, overpayment
from tests.helpers import make_model, ripen


def run(model, steps):
    for _ in range(steps):
        if not model.running:
            break
        model.step()
    return model


PROPS = [
    {"picker": 1, "bid": 12.0, "cost": 10.0},
    {"picker": 2, "bid": 9.0, "cost": 9.0},
    {"picker": 3, "bid": 15.0, "cost": 8.0},
]


class TestSealedBid(unittest.TestCase):
    def test_first_price_pays_own_bid(self):
        win, pay = first_price(PROPS)
        self.assertEqual((win["picker"], pay), (2, 9.0))

    def test_second_price_pays_second_lowest_bid(self):
        win, pay = second_price(PROPS)
        self.assertEqual((win["picker"], pay), (2, 12.0))

    def test_second_price_single_bidder_pays_own_bid(self):
        win, pay = second_price(PROPS[:1])
        self.assertEqual(pay, 12.0)

    def test_regret_measures_misallocation(self):
        # picker 2 wins (lowest bid) but picker 3 had the lowest true cost
        self.assertEqual(regret(9.0, PROPS), 1.0)


class TestEnglishClock(unittest.TestCase):
    def test_lowest_cost_wins_near_second_lowest_cost(self):
        costs = {1: 30.0, 2: 12.0, 3: 20.0, 4: None}
        winner, price, rounds, replies = english_clock(costs, 120.0, 4.0, random.Random(0))
        self.assertEqual(winner, 2)
        self.assertGreaterEqual(price, 20.0)          # at least the runner-up's cost
        self.assertLess(price, 20.0 + 4.0)            # within one decrement of it
        self.assertGreater(rounds, 1)
        self.assertGreaterEqual(replies, len(costs))

    def test_nobody_accepts_opening_price(self):
        winner, _, _, _ = english_clock({1: 500.0, 2: None}, 120.0, 4.0, random.Random(0))
        self.assertIsNone(winner)

    def test_sole_bidder_wins_at_opening_price(self):
        winner, price, _, _ = english_clock({1: 10.0, 2: None}, 120.0, 4.0, random.Random(0))
        self.assertEqual((winner, price), (1, 120.0))


class TestWinnerDetermination(unittest.TestCase):
    def test_bundle_beats_two_single_bids_when_cheaper(self):
        bids = [
            {"picker": 1, "areas": ("a",), "trees": 3, "bid": 20.0, "cost": 20.0},
            {"picker": 2, "areas": ("b",), "trees": 3, "bid": 20.0, "cost": 20.0},
            {"picker": 1, "areas": ("a", "b"), "trees": 6, "bid": 25.0, "cost": 25.0},
        ]
        winners = solve_winner_determination(bids, value_per_tree=10.0)
        self.assertEqual([w["areas"] for w in winners], [("a", "b")])

    def test_each_area_and_picker_used_once(self):
        bids = [
            {"picker": 1, "areas": ("a",), "trees": 3, "bid": 5.0, "cost": 5.0},
            {"picker": 1, "areas": ("b",), "trees": 3, "bid": 5.0, "cost": 5.0},
            {"picker": 2, "areas": ("a",), "trees": 3, "bid": 6.0, "cost": 6.0},
        ]
        winners = solve_winner_determination(bids, value_per_tree=10.0)
        self.assertEqual(len(winners), 2)
        self.assertEqual(len({w["picker"] for w in winners}), 2)
        self.assertEqual(sorted(a for w in winners for a in w["areas"]), ["a", "b"])

    def test_unprofitable_bids_are_not_accepted(self):
        bids = [{"picker": 1, "areas": ("a",), "trees": 1, "bid": 50.0, "cost": 50.0}]
        self.assertEqual(solve_winner_determination(bids, value_per_tree=10.0), [])


class TestStrategicBidding(unittest.TestCase):
    def setUp(self):
        self.model = make_model(mode=NOVEL, strategic_bidders=1.0, n_pickers=2)
        self.picker = self.model.pickers[0]

    def test_markup_applies_in_first_price(self):
        self.assertGreater(self.picker.markup, 0)
        self.assertGreater(self.picker.bid_for(10.0), 10.0)

    def test_truthful_in_vickrey(self):
        self.model.allocation = VICKREY
        self.assertEqual(self.picker.bid_for(10.0), 10.0)

    def test_proposal_carries_true_cost(self):
        tree = max(self.model.trees_by_pos)          # far from the base, so the cost is > 0
        ripen(self.model, tree)
        drone = self.model.drones[0]
        content = {"patch": 0, "trees": [tree], "urgency": 0.5, "task_id": 99}
        self.model.bus.send(msg.CFP, drone.unique_id, self.picker.unique_id, content)
        self.picker.handle_messages()
        reply = self.model.bus.receive(drone.unique_id)[0]
        self.assertEqual(reply.performative, msg.PROPOSE)
        self.assertGreater(reply.content["cost"], reply.content["true_cost"])


class TestBundleBids(unittest.TestCase):
    def test_bundles_respect_spare_capacity(self):
        model = make_model(mode=NOVEL, picker_capacity=20)
        picker = model.pickers[0]
        areas = [{"patch": pid, "trees": [t.pos for t in trees], "urgency": 0.5}
                 for pid, trees in list(model.patches.items())[:3]]
        bids = picker.bundle_bids(areas)
        self.assertTrue(bids)
        self.assertTrue(any(len(b["areas"]) == 2 for b in bids))     # 9 + 9 trees fit in 20
        self.assertTrue(all(b["trees"] <= picker.spare_capacity for b in bids))
        single = {b["areas"][0]: b["cost"] for b in bids if len(b["areas"]) == 1}
        for b in bids:
            if len(b["areas"]) == 2:          # a tour through two areas costs at least each alone
                self.assertGreaterEqual(b["cost"], max(single[a] for a in b["areas"]))


class TestAuctionModesRun(unittest.TestCase):
    def test_every_auction_awards_and_completes_contracts(self):
        for allocation in (CONTRACT_NET, VICKREY, ENGLISH, COMBINATORIAL):
            with self.subTest(allocation=allocation):
                model = run(FarmModel(mode=NOVEL, allocation=allocation, seed=5), 400)
                self.assertGreater(model.contracts_awarded, 0)
                self.assertGreater(model.picked, 0)
                self.assertTrue(all(p.is_active for p in model.pickers))

    def test_vickrey_is_efficient_with_strategic_bidders(self):
        model = run(FarmModel(mode=NOVEL, allocation=VICKREY, strategic_bidders=0.5, seed=4), 400)
        self.assertEqual(mean_regret(model), 0.0)
        self.assertGreaterEqual(overpayment(model), 1.0)

    def test_english_auction_costs_more_messages(self):
        cnp = run(FarmModel(mode=NOVEL, seed=6), 300)
        eng = run(FarmModel(mode=NOVEL, allocation=ENGLISH, seed=6), 300)
        self.assertGreater(eng.bus.sent, cnp.bus.sent)
        self.assertGreater(eng.english_rounds, 0)

    def test_combinatorial_solves_winner_determination(self):
        model = run(FarmModel(mode=NOVEL, allocation=COMBINATORIAL, seed=5), 300)
        self.assertGreater(model.wdp_solves, 0)


if __name__ == "__main__":
    unittest.main()
