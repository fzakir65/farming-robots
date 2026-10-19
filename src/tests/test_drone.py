"""Unit tests for the explorer drone."""
import unittest

from farm import messages as msg
from farm.constants import BASIC, CRUISE_ALTITUDE, DIRECT, DRONE_BATTERY, EXTENDED, NOVEL, TREE_HEIGHT
from farm.drone import EXPLORING, GOING_TO_CROP, LANDED, RETURNING, WAITING
from tests.helpers import first_tree, lane_cell_next_to_tree, make_model, place, ripen


def airborne(model, drone, pos, altitude=CRUISE_ALTITUDE):
    place(model, drone, pos)
    drone.altitude = altitude
    drone.state = EXPLORING


class TestDroneFlight(unittest.TestCase):
    def setUp(self):
        self.model = make_model(mode=BASIC)
        self.drone = self.model.drones[0]

    def test_takes_off_and_climbs_to_cruise_altitude(self):
        self.assertEqual(self.drone.make_decision(), "ascend")
        for _ in range(CRUISE_ALTITUDE):
            self.drone.step()
        self.assertEqual(self.drone.altitude, CRUISE_ALTITUDE)
        self.assertEqual(self.drone.state, EXPLORING)

    def test_six_movement_directions(self):
        airborne(self.model, self.drone, (5, 15))
        start = self.drone.pos
        self.drone.move_front()
        self.assertEqual(self.drone.pos, (start[0], start[1] + 1))
        self.drone.move_back()
        self.drone.move_right()
        self.assertEqual(self.drone.pos, (start[0] + 1, start[1]))
        self.drone.move_left()
        self.assertEqual(self.drone.pos, start)
        self.drone.ascend()
        self.assertEqual(self.drone.altitude, CRUISE_ALTITUDE + 1)
        self.drone.descend()
        self.assertEqual(self.drone.altitude, CRUISE_ALTITUDE)

    def test_must_climb_above_trees(self):
        tree = first_tree(self.model)
        lane = (tree[0] - 1, tree[1])
        airborne(self.model, self.drone, lane, altitude=TREE_HEIGHT)
        self.assertFalse(self.drone.move_right())
        self.assertEqual(self.drone.pos, lane)
        self.assertEqual(self.drone.altitude, TREE_HEIGHT + 1)
        self.assertTrue(self.drone.move_right())
        self.assertEqual(self.drone.pos, tree)

    def test_cannot_descend_into_trees(self):
        airborne(self.model, self.drone, first_tree(self.model), altitude=TREE_HEIGHT + 1)
        self.drone.descend()
        self.assertEqual(self.drone.altitude, TREE_HEIGHT + 1)

    def test_camera_range_grows_with_altitude(self):
        tree = ripen(self.model, first_tree(self.model))
        airborne(self.model, self.drone, (tree.pos[0] + 3, tree.pos[1]), altitude=2)
        self.assertNotIn(tree, self.drone.scan())
        self.drone.altitude = 3
        self.assertIn(tree, self.drone.scan())

    def test_returns_lands_and_charges_when_battery_low(self):
        airborne(self.model, self.drone, (10, 10))
        self.drone.battery = 8
        self.assertEqual(self.drone.make_decision(), "return_to_pad")
        self.drone.battery = DRONE_BATTERY      # enough to finish the trip in this test
        for _ in range(60):
            self.drone.return_to_pad()
            if self.drone.state == LANDED:
                break
        self.assertEqual(self.drone.state, LANDED)
        self.assertIn(self.drone.pos, self.model.farm.drone_pad)
        self.drone.battery = 50
        self.assertEqual(self.drone.make_decision(), "charge")


class TestBasicModeGuarding(unittest.TestCase):
    def setUp(self):
        self.model = make_model(mode=BASIC)
        self.drone = self.model.drones[0]
        self.tree = ripen(self.model, first_tree(self.model))
        airborne(self.model, self.drone, (self.tree.pos[0] + 2, self.tree.pos[1]))

    def fly_until_waiting(self):
        self.drone.step()
        for _ in range(10):
            if self.drone.state == WAITING:
                return
            self.drone.fly_to_crop()
        self.fail("drone never reached the crop")

    def test_flies_to_crop_and_waits(self):
        self.drone.findings = self.drone.scan()
        self.assertEqual(self.drone.make_decision(), "fly_to_crop")
        self.assertEqual(self.drone.state, GOING_TO_CROP)
        self.fly_until_waiting()
        self.assertEqual(self.drone.pos, self.tree.pos)
        self.assertIn(self.tree.patch_id, self.model.guarded_patches)

    def test_resumes_exploring_when_picker_arrives(self):
        self.fly_until_waiting()
        place(self.model, self.model.pickers[0], lane_cell_next_to_tree(self.model, self.tree.pos))
        self.drone.wait_for_picker()
        self.assertEqual(self.drone.state, EXPLORING)
        self.assertNotIn(self.tree.patch_id, self.model.guarded_patches)

    def test_second_drone_does_not_guard_same_area(self):
        other = make_model(mode=BASIC, n_drones=2)
        tree = ripen(other, first_tree(other))
        d1, d2 = other.drones
        for d in (d1, d2):
            airborne(other, d, (tree.pos[0] + 2, tree.pos[1]))
            d.findings = d.scan()
        self.assertTrue(d1.choose_crop_to_guard())
        self.assertFalse(d2.choose_crop_to_guard())


class TestExtendedModeSignals(unittest.TestCase):
    def test_broadcast_reaches_every_picker_once(self):
        model = make_model(mode=EXTENDED, n_pickers=3)
        drone = model.drones[0]
        tree = ripen(model, first_tree(model))
        drone.report_crops([tree])
        drone.report_crops([tree])         # same area again: suppressed
        for p in model.pickers:
            inbox = model.bus.receive(p.unique_id)
            self.assertEqual(len(inbox), 1)
            self.assertEqual(inbox[0].performative, msg.INFORM_CROP)
            self.assertIn(tree.pos, inbox[0].content["trees"])
        self.assertEqual(model.bus.sent, 1)

    def test_direct_message_goes_to_nearest_free_picker(self):
        model = make_model(mode=EXTENDED, comm_mode=DIRECT, n_pickers=2)
        drone = model.drones[0]
        tree = ripen(model, first_tree(model))
        near, far = model.pickers
        place(model, near, lane_cell_next_to_tree(model, tree.pos))
        place(model, far, (model.farm.width - 1, model.farm.height - 1))
        drone.report_crops([tree])
        self.assertEqual(len(model.bus.receive(near.unique_id)), 1)
        self.assertEqual(len(model.bus.receive(far.unique_id)), 0)


class TestContractNetManager(unittest.TestCase):
    def test_full_protocol_awards_cheapest_bid(self):
        model = make_model(mode=NOVEL, n_pickers=2)
        drone = model.drones[0]
        tree = ripen(model, first_tree(model))
        near, far = model.pickers
        place(model, near, lane_cell_next_to_tree(model, tree.pos))
        place(model, far, (model.farm.width - 1, model.farm.height - 1))

        drone.call_for_proposals([tree])               # 1. CFP to all pickers
        self.assertEqual(len(drone.pending_cfps), 1)
        for p in (near, far):
            p.handle_messages()                        # 2. each picker bids
        model.schedule.steps += 2                      # 3. deadline passes
        drone.handle_messages()
        drone.award_contracts()                        # 4. award + reject

        near.handle_messages()
        far_inbox = model.bus.receive(far.unique_id)
        self.assertIn(tree.pos, near.tasks)
        self.assertNotIn(tree.pos, far.tasks)
        self.assertEqual([m.performative for m in far_inbox], [msg.REJECT])
        self.assertEqual(model.contracts_awarded, 1)
        self.assertEqual(drone.pending_cfps, {})

    def test_no_bids_means_task_is_announced_again_later(self):
        model = make_model(mode=NOVEL, n_pickers=1)
        drone, picker = model.drones[0], model.pickers[0]
        picker.load = picker.capacity                  # cannot accept anything
        tree = ripen(model, first_tree(model))
        drone.call_for_proposals([tree])
        picker.handle_messages()
        model.schedule.steps += 2
        drone.handle_messages()
        drone.award_contracts()
        self.assertEqual(model.contracts_awarded, 0)
        model.schedule.steps += 10
        drone.call_for_proposals([tree])
        self.assertEqual(len(drone.pending_cfps), 1)


if __name__ == "__main__":
    unittest.main()
