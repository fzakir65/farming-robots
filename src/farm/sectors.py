"""Game theory for the drones: which part of the farm should each drone search?

The farm is split into sectors. A drone searching sector *s* expects to find
``v_s / n_s`` new ripe crops, where ``v_s`` is how productive the sector is and
``n_s`` is how many drones search it. Drones that pick the same sector share its
crops. This is a **congestion game**, so it is also a *potential game* (Rosenthal,
1973; Monderer & Shapley, 1996): it always has a pure-strategy Nash equilibrium,
and best-response dynamics is guaranteed to reach one.

Two strategies use this (``drone_strategy`` in the model):

* **Nash sector game:** every drone shares its sector values. Each drone computes
  the same equilibrium and searches the sector it is assigned.
* **Fictitious play** (Brown, 1951): every drone keeps its *own* value estimates
  and the empirical frequency of the other drones' choices. It best-responds to the
  expected congestion. This is the learning approach of WS6 (follow what has worked).

``mixed_equilibria_2p`` uses nashpy to list every equilibrium (pure and mixed) of
the 2-drone game, which the tests and the analysis notebook use as a cross-check.
"""
import numpy as np


def make_sectors(farm):
    """Split the farm into 4 sectors: left/right of the river x bottom/top half."""
    river_x = sorted({x for x, _ in farm.water_cells})
    split_x = river_x[len(river_x) // 2] if river_x else farm.width // 2
    split_y = farm.height // 2
    xs = [(0, split_x - 1), (split_x, farm.width - 1)]
    ys = [(0, split_y - 1), (split_y, farm.height - 1)]
    return [(x0, x1, y0, y1) for (y0, y1) in ys for (x0, x1) in xs]


def sector_of(sectors, pos):
    for i, (x0, x1, y0, y1) in enumerate(sectors):
        if x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1:
            return i
    return 0


def sector_centre(sector):
    x0, x1, y0, y1 = sector
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def payoff(values, choices, i):
    """Expected new crops for drone ``i`` given every drone's sector choice."""
    s = choices[i]
    return values[s] / sum(1 for c in choices if c == s)


def potential(values, choices):
    """Rosenthal's potential: sum over sectors of v_s * (1 + 1/2 + ... + 1/n_s)."""
    total = 0.0
    for s, v in enumerate(values):
        n = sum(1 for c in choices if c == s)
        total += v * sum(1.0 / k for k in range(1, n + 1))
    return total


def best_response(values, others):
    """Sector that maximises v_s / (1 + drones already there)."""
    return max(range(len(values)), key=lambda s: (values[s] / (1 + others.count(s)), -s))


def is_nash(values, choices, tol=1e-9):
    """True if no drone can gain by switching sector on its own."""
    for i in range(len(choices)):
        current = payoff(values, choices, i)
        others = choices[:i] + choices[i + 1:]
        for s in range(len(values)):
            if values[s] / (1 + others.count(s)) > current + tol:
                return False
    return True


def nash_assignment(values, n_drones, max_rounds=100):
    """
    A pure Nash equilibrium found by best-response dynamics.

    Every improving move raises the potential, so the loop always ends.
    The start (everyone in the best sector) and the order are fixed, so every drone
    computes the same answer independently.
    """
    choices = [max(range(len(values)), key=lambda s: (values[s], -s))] * n_drones
    for _ in range(max_rounds):
        changed = False
        for i in range(n_drones):
            others = choices[:i] + choices[i + 1:]
            br = best_response(values, others)
            if values[br] / (1 + others.count(br)) > payoff(values, choices, i) + 1e-9:
                choices[i] = br
                changed = True
        if not changed:
            break
    return choices


def mixed_equilibria_2p(values):
    """All Nash equilibria (pure and mixed) of the 2-drone sector game, via nashpy."""
    import nashpy

    k = len(values)
    a = np.array([[values[i] / (2 if i == j else 1) for j in range(k)] for i in range(k)], dtype=float)
    game = nashpy.Game(a, a.T)
    return [(np.round(p, 4), np.round(q, 4)) for p, q in game.support_enumeration()]


class FictitiousPlayer:
    """One drone's beliefs: its own sector values and how often others pick each sector."""

    def __init__(self, n_sectors, prior):
        self.values = list(prior)
        self.counts = [1.0] * n_sectors     # Laplace-smoothed frequencies of others' choices
        self.n_sectors = n_sectors

    def observe_choice(self, sector):
        self.counts[sector] += 1

    def observe_value(self, sector, found, rate):
        self.values[sector] = (1 - rate) * self.values[sector] + rate * found

    def choose(self, n_others):
        total = sum(self.counts)
        expected = [n_others * c / total for c in self.counts]
        return max(range(self.n_sectors), key=lambda s: (self.values[s] / (1 + expected[s]), -s))
