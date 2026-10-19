"""Shared helpers for building small, controlled test scenarios."""
from farm.constants import BASIC
from farm.environment import RIPE
from farm.model import FarmModel


def make_model(mode=BASIC, ripe_patches=0, n_pickers=2, n_drones=1, **kwargs):
    """A model with no ripe crops (unless asked) so each test sets up its own scene."""
    return FarmModel(
        mode=mode,
        initial_ripe_patches=ripe_patches,
        n_pickers=n_pickers,
        n_drones=n_drones,
        seed=42,
        **kwargs,
    )


def ripen(model, pos):
    tree = model.tree_at(pos)
    tree.ripen()
    assert tree.state == RIPE
    return tree


def lane_cell_next_to_tree(model, tree_pos):
    """A walkable cell right beside ``tree_pos`` (distance 1)."""
    x, y = tree_pos
    for cell in [(x - 1, y), (x + 1, y), (x - 2, y), (x + 2, y)]:
        if model.farm.is_passable(cell) and not model.farm.is_water(cell):
            return cell
    raise AssertionError(f"no lane next to {tree_pos}")


def first_tree(model):
    """Bottom-left tree of the first row (it has a lane on its left)."""
    return min(model.trees_by_pos, key=lambda p: (p[0], p[1]))


def place(model, agent, pos):
    model.grid.move_agent(agent, pos)
