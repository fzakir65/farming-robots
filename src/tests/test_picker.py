"""Unit tests for the amphibious picker robot."""
import unittest

from farm import messages as msg
from farm.constants import BASIC, EXTENDED, NOVEL, PICKER_BATTERY, WATER_SLOWDOWN
from farm.environment import EMPTY
from farm.picker import (
    AT_BASE,
    CHARGING,
    MOVING_TO_CROP,
    OUT_OF_BATTERY,
    PICKING,
    RETURNING,
    STANDOFF_LIMIT,
)
from tests.helpers import first_tree, lane_cell_next_to_tree, make_model, place, ripen


class TestPickerDecisions(unittest.TestCase):
    def setUp(self):
        self.model = make_model(mode=BASIC)
        self.picker = self.model.pickers[0]
        self.tree_pos = first_tree(self.model)
        self.lane = lane_cell_next_to_tree(self.model, self.tree_pos)

    def test_basic_mode_leaves_base_to_explore(self):
        self.assertEqual(self.picker.make_decision(), "explore")

    def test_picks_ripe_crop_within_reach(self):
        ripen(self.model, self.tree_pos)
        place(self.model, self.picker, self.lane)
        self.assertEqual(self.picker.make_decision(), "pick")
        self.picker.pick()
        self.assertEqual(self.picker.load, 1)
        self.assertEqual(self.picker.state, PICKING)
        self.assertEqual(self.model.tree_at(self.tree_pos).state, EMPTY)

    def test_approaches_ripe_crop_in_sight(self):
        ripen(self.model, self.tree_pos)
        x, y = self.lane
        place(self.model, self.picker, (x, y + 3))
        self.picker.state = "exploring"
        self.assertEqual(self.picker.make_decision(), "approach_crop")
        self.picker.approach_crop()
        self.assertEqual(self.picker.state, MOVING_TO_CROP)
        self.assertEqual(self.picker.target, self.tree_pos)

    def test_returns_when_storage_is_full(self):
        place(self.model, self.picker, self.lane)
        self.picker.load = self.picker.capacity
        self.assertEqual(self.picker.make_decision(), "return_to_base")

    def test_returns_when_battery_is_low(self):
        place(self.model, self.picker, self.lane)
        self.picker.battery = self.picker.energy_to_base()
        self.assertEqual(self.picker.make_decision(), "return_to_base")

    def test_unloads_at_base(self):
        self.picker.load = 4
        self.assertEqual(self.picker.make_decision(), "unload")
        self.picker.unload()
        self.assertEqual(self.picker.load, 0)
        self.assertEqual(self.model.delivered, 4)
        self.assertEqual(self.picker.state, AT_BASE)

    def test_charges_at_base(self):
        self.picker.battery = 50
        self.assertEqual(self.picker.make_decision(), "charge")
        self.picker.charge()
        self.assertEqual(self.picker.state, CHARGING)
        self.assertGreater(self.picker.battery, 50)
        for _ in range(100):
            self.picker.charge()
        self.assertEqual(self.picker.battery, PICKER_BATTERY)


class TestPickerMovement(unittest.TestCase):
    def setUp(self):
        self.model = make_model(mode=BASIC)
        self.picker = self.model.pickers[0]

    def test_never_drives_through_trees(self):
        for _ in range(300):
            self.picker.explore()
            self.assertNotIn(self.picker.pos, self.model.farm.tree_cells)

    def test_slowed_down_in_water(self):
        water = min(self.model.farm.water_cells)
        place(self.model, self.picker, water)
        nxt = (water[0] + 1, water[1])
        moved = [self.picker.step_to(nxt) for _ in range(WATER_SLOWDOWN)]
        self.assertEqual(moved, [False] * (WATER_SLOWDOWN - 1) + [True])
        self.assertEqual(self.picker.pos, nxt)

    def test_not_slowed_on_land(self):
        start = self.model.farm.picker_base[0]
        self.assertTrue(self.picker.step_to((start[0], start[1] + 1)))

    def test_moving_uses_battery(self):
        before = self.picker.battery
        self.picker.explore()
        self.assertLess(self.picker.battery, before)

    def test_return_to_base_reaches_base(self):
        place(self.model, self.picker, (self.model.farm.width - 1, self.model.farm.height - 1))
        for _ in range(200):
            self.picker.return_to_base()
            if self.picker.at_base:
                break
        self.assertTrue(self.picker.at_base)
        self.assertEqual(self.picker.state, RETURNING)

    def test_waits_behind_another_picker(self):
        other = self.model.pickers[1]
        place(self.model, self.picker, (0, 5))
        place(self.model, other, (0, 6))
        self.picker.goals = {(0, 10)}
        self.picker.path = [(0, 6), (0, 7)]
        self.picker.follow_path()
        self.assertEqual(self.picker.pos, (0, 5))
        self.assertEqual(self.picker.blocked_steps, 1)

    def test_squeezes_past_when_making_no_progress(self):
        other = self.model.pickers[1]
        place(self.model, self.picker, (0, 5))
        place(self.model, other, (0, 6))
        self.picker.path = [(0, 6), (0, 7)]
        self.picker.no_progress = STANDOFF_LIMIT      # e.g. after dodging back and forth
        self.picker.follow_path()
        self.assertEqual(self.picker.pos, (0, 6))

    def test_no_progress_counter_survives_replanning(self):
        place(self.model, self.picker, (0, 5))
        self.picker.goals = {(0, 10)}
        self.picker.plan()
        self.picker.track_progress()
        self.picker.no_progress = 5
        self.picker.plan(avoid_pickers=True)        # same destination
        self.assertEqual(self.picker.no_progress, 5)

    def test_runs_out_of_battery(self):
        self.picker.use_energy(PICKER_BATTERY + 1)
        self.assertEqual(self.picker.state, OUT_OF_BATTERY)
        pos = self.picker.pos
        self.picker.step()
        self.assertEqual(self.picker.pos, pos)


class TestPickerCommunication(unittest.TestCase):
    def setUp(self):
        self.model = make_model(mode=EXTENDED)
        self.picker = self.model.pickers[0]
        self.tree_pos = first_tree(self.model)

    def test_waits_at_base_without_a_signal(self):
        self.assertEqual(self.picker.make_decision(), "wait_at_base")

    def test_leaves_base_when_drone_reports_crops(self):
        ripen(self.model, self.tree_pos)
        self.model.bus.send(msg.INFORM_CROP, 99, self.picker.unique_id,
                            {"trees": [self.tree_pos], "urgency": 0.4, "patch": 0})
        self.picker.handle_messages()
        self.assertIn(self.tree_pos, self.picker.tasks)
        self.assertEqual(self.picker.make_decision(), "go_to_next_task")
        self.picker.go_to_next_task()
        self.assertEqual(self.picker.state, MOVING_TO_CROP)
        self.assertEqual(self.picker.target, self.tree_pos)

    def test_does_not_set_off_without_enough_battery_for_the_round_trip(self):
        far_tree = max(self.model.trees_by_pos, key=lambda p: p[0] + p[1])
        ripen(self.model, far_tree)
        place(self.model, self.picker, (0, 1))           # just outside the base
        self.picker.battery = 20
        self.picker.tasks[far_tree] = {"urgency": 0.5, "task_id": None, "manager": None}
        self.picker.go_to_next_task()
        self.assertTrue(self.picker.needs_recharge)
        self.assertEqual(self.picker.state, RETURNING)

    def test_forgets_task_when_crop_is_gone(self):
        self.picker.tasks[self.tree_pos] = {"urgency": 0.5, "task_id": None, "manager": None}
        place(self.model, self.picker, lane_cell_next_to_tree(self.model, self.tree_pos))
        self.picker.observe()          # the tree is not ripe
        self.assertNotIn(self.tree_pos, self.picker.tasks)

    def test_reports_completed_contract_to_manager(self):
        model = make_model(mode=NOVEL)
        picker, drone = model.pickers[0], model.drones[0]
        tree = ripen(model, first_tree(model))
        picker.tasks[tree.pos] = {"urgency": 0.5, "task_id": 7, "manager": drone.unique_id}
        place(model, picker, lane_cell_next_to_tree(model, tree.pos))
        picker.pick()
        inbox = model.bus.receive(drone.unique_id)
        self.assertEqual([m.performative for m in inbox], [msg.INFORM_DONE])
        self.assertEqual(inbox[0].content["task_id"], 7)


class TestContractNetBids(unittest.TestCase):
    def setUp(self):
        self.model = make_model(mode=NOVEL)
        self.tree_pos = first_tree(self.model)
        self.task = {"task_id": 1, "trees": [self.tree_pos], "urgency": 0.5, "patch": 0}

    def test_closer_picker_bids_lower(self):
        near, far = self.model.pickers
        place(self.model, near, lane_cell_next_to_tree(self.model, self.tree_pos))
        place(self.model, far, (self.model.farm.width - 1, self.model.farm.height - 1))
        self.assertLess(near.estimate_cost(self.task), far.estimate_cost(self.task))

    def test_busy_picker_bids_higher(self):
        picker = self.model.pickers[0]
        idle_bid = picker.estimate_cost(self.task)
        # already committed to a crop at the same place: same travel, but a longer queue
        picker.tasks[self.tree_pos] = {"urgency": 0, "task_id": 5, "manager": 0}
        self.assertGreater(picker.estimate_cost(self.task), idle_bid)

    def test_full_picker_refuses(self):
        picker = self.model.pickers[0]
        picker.load = picker.capacity
        self.assertIsNone(picker.estimate_cost(self.task))
        drone_id = self.model.drones[0].unique_id
        picker.respond_to_cfp(msg.Message(msg.CFP, drone_id, None, self.task))
        self.assertEqual(self.model.bus.receive(drone_id)[0].performative, msg.REFUSE)

    def test_accept_adds_contracted_task(self):
        picker = self.model.pickers[0]
        self.model.bus.send(msg.ACCEPT, self.model.drones[0].unique_id, picker.unique_id, self.task)
        picker.handle_messages()
        self.assertEqual(picker.tasks[self.tree_pos]["task_id"], 1)


if __name__ == "__main__":
    unittest.main()
