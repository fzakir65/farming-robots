"""Unit tests for environment creation, path finding and the crop life-cycle."""
import unittest

from farm.constants import (
    BASIC,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    EXTENDED,
    PATCH_LENGTH,
    RIPE_AGE,
    RIVER_WIDTH,
    ROT_AGE,
    TREE_ROW_WIDTH,
)
from farm.drone import ExplorerDrone
from farm.environment import EMPTY, GROWING, RIPE, Base, Ground, Tree, Water
from farm.model import FarmModel, ripe_on_field
from farm.picker import PickerRobot
from tests.helpers import first_tree, make_model, ripen


class TestEnvironmentCreation(unittest.TestCase):
    def setUp(self):
        self.model = make_model(n_pickers=3, n_drones=2)

    def test_grid_size(self):
        self.assertEqual(self.model.grid.width, DEFAULT_WIDTH)
        self.assertEqual(self.model.grid.height, DEFAULT_HEIGHT)

    def test_every_cell_has_exactly_one_terrain_agent(self):
        terrain = (Ground, Water, Base, Tree)
        for x in range(self.model.grid.width):
            for y in range(self.model.grid.height):
                here = [a for a in self.model.grid.get_cell_list_contents([(x, y)]) if isinstance(a, terrain)]
                self.assertEqual(len(here), 1, f"cell {(x, y)}")

    def test_river_crosses_the_whole_field(self):
        water = [a for content, _ in self.model.grid.coord_iter() for a in content if isinstance(a, Water)]
        self.assertEqual(len(water), RIVER_WIDTH * DEFAULT_HEIGHT)
        self.assertEqual({a.pos[1] for a in water}, set(range(DEFAULT_HEIGHT)))

    def test_trees_are_organised_in_rows_of_three(self):
        columns = sorted({p[0] for p in self.model.trees_by_pos})
        self.assertEqual(len(columns) % TREE_ROW_WIDTH, 0)
        self.assertEqual(len(columns) // TREE_ROW_WIDTH, 6)

    def test_crop_areas_are_at_most_3x3(self):
        for trees in self.model.patches.values():
            self.assertLessEqual(len(trees), TREE_ROW_WIDTH * PATCH_LENGTH)

    def test_every_tree_can_be_reached_by_a_picker(self):
        farm = self.model.farm
        for pos in self.model.trees_by_pos:
            reachable = [p for p in farm.pick_positions(pos) if p in farm.base_distance]
            self.assertTrue(reachable, f"tree at {pos} cannot be reached")

    def test_robots_start_at_their_bases(self):
        self.assertEqual(len(self.model.pickers), 3)
        self.assertEqual(len(self.model.drones), 2)
        for p in self.model.pickers:
            self.assertIn(p.pos, self.model.farm.picker_base)
        for d in self.model.drones:
            self.assertIn(d.pos, self.model.farm.drone_pad)
            self.assertEqual(d.altitude, 0)

    def test_only_robots_are_scheduled(self):
        for agent in self.model.schedule.agents:
            self.assertIsInstance(agent, (PickerRobot, ExplorerDrone))
        self.assertEqual(self.model.schedule.get_agent_count(), 5)

    def test_initial_ripe_patches(self):
        model = make_model(ripe_patches=4)
        ripe_patches = {t.patch_id for t in model.trees if t.is_ripe}
        self.assertEqual(len(ripe_patches), 4)
        self.assertEqual(ripe_on_field(model), sum(len(model.patches[p]) for p in ripe_patches))

    def test_random_layouts_differ_but_stay_valid(self):
        a = FarmModel(random_layout=True, seed=1)
        b = FarmModel(random_layout=True, seed=2)
        self.assertNotEqual(set(a.trees_by_pos), set(b.trees_by_pos))
        for model in (a, b):
            for pos in model.trees_by_pos:
                self.assertTrue(any(p in model.farm.base_distance for p in model.farm.pick_positions(pos)))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            FarmModel(mode="Turbo")


class TestPathFinding(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.farm = self.model.farm

    def test_path_never_goes_through_trees(self):
        start = self.farm.picker_base[0]
        goal = (self.farm.width - 1, self.farm.height - 1)
        path = self.farm.shortest_path(start, {goal})
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], goal)
        for cell in path:
            self.assertNotIn(cell, self.farm.tree_cells)

    def test_path_moves_one_cell_at_a_time(self):
        path = [self.farm.picker_base[0]] + self.farm.shortest_path(
            self.farm.picker_base[0], {(self.farm.width - 1, 0)}
        )
        for a, b in zip(path, path[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_crossing_water_costs_more_than_land(self):
        river_x = min(x for x, _ in self.farm.water_cells)
        y = 0
        left, right = (river_x - 1, y), (river_x + 2, y)
        cost = self.farm.distance_map([left])[right]
        self.assertEqual(cost, 1 + 2 * 2)   # bank -> water (1), water -> water (2), water -> bank (2)

    def test_unreachable_goal_returns_none(self):
        tree = first_tree(self.model)
        self.assertIsNone(self.farm.shortest_path(self.farm.picker_base[0], {tree}))


class TestCropLifeCycle(unittest.TestCase):
    def test_tree_ripens_then_rots(self):
        model = make_model(mode=EXTENDED)
        tree = model.tree_at(first_tree(model))
        tree.replant(0)
        for _ in range(RIPE_AGE):
            tree.grow()
        self.assertEqual(tree.state, RIPE)
        wasted = model.wasted
        for _ in range(ROT_AGE - RIPE_AGE):
            tree.grow()
        self.assertEqual(tree.state, EMPTY)
        self.assertEqual(model.wasted, wasted + 1)

    def test_urgency_grows_as_crop_ages(self):
        model = make_model(mode=EXTENDED)
        tree = ripen(model, first_tree(model))
        early = tree.urgency
        tree.age = (RIPE_AGE + ROT_AGE) // 2
        self.assertGreater(tree.urgency, early)
        self.assertLessEqual(tree.urgency, 1.0)

    def test_harvest_only_ripe(self):
        model = make_model()
        tree = model.tree_at(first_tree(model))
        self.assertFalse(tree.harvest())
        ripen(model, tree.pos)
        self.assertTrue(tree.harvest())
        self.assertEqual(tree.state, EMPTY)
        self.assertEqual(model.picked, 1)

    def test_fully_picked_area_is_replanted(self):
        model = make_model(mode=EXTENDED)
        patch_id, trees = next(iter(model.patches.items()))
        for t in trees:
            t.state = EMPTY
        model.grow_crops()
        self.assertEqual(model.replanted, 1)
        self.assertTrue(all(t.state in (GROWING, EMPTY) and t.age <= 0 for t in trees))

    def test_basic_mode_crops_do_not_age(self):
        model = make_model(mode=BASIC, ripe_patches=2)
        ages = [t.age for t in model.trees]
        for _ in range(20):
            model.step()
        self.assertEqual(ages, [t.age for t in model.trees])


if __name__ == "__main__":
    unittest.main()
