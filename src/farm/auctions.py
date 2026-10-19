"""Auction mechanisms used by the drones (managers) in the novel mode.

Pickers are *sellers* of a service (picking a crop area), so these are **reverse**
(procurement) auctions: the lowest bid wins, and the winner is *paid*.

* **First-price sealed bid** (the Contract Net award rule): the winner is paid its own
  bid. A self-interested picker gains by bidding above its true cost (bid shading).
* **Vickrey / second-price sealed bid** (Vickrey, 1961): the winner is paid the
  *second*-lowest bid. Bidding the true cost is a dominant strategy.
* **English / descending clock**: the price falls each round, and pickers leave once
  it drops below their cost. The last one left wins at the current price. With
  truthful dropping, this matches Vickrey's outcome (the revenue equivalence theorem).
* **Combinatorial**: pickers bid on *bundles* of areas, because one trip can serve
  several nearby areas. The manager solves the winner-determination problem exactly
  with OR-Tools CP-SAT.

Each proposal is a dict ``{"picker", "bid", "cost"}``. ``cost`` is the bidder's true
cost. The manager never uses it; it is kept only to measure allocation regret.
"""
from ortools.sat.python import cp_model


def first_price(proposals):
    """Lowest bid wins and is paid its own bid."""
    win = min(proposals, key=lambda p: (p["bid"], p["picker"]))
    return win, win["bid"]


def second_price(proposals):
    """Lowest bid wins and is paid the second-lowest bid (its own if it is alone)."""
    ordered = sorted(proposals, key=lambda p: (p["bid"], p["picker"]))
    win = ordered[0]
    return win, (ordered[1]["bid"] if len(ordered) > 1 else win["bid"])


def english_clock(costs, start, decrement, rng):
    """
    Descending-clock reverse auction.

    ``costs`` maps picker id to its reservation cost (None = cannot take the task).
    A picker stays in while the price is at least its cost. Returns
    ``(winner, price, rounds, replies)``, with ``winner = None`` if nobody accepts the
    opening price. ``replies`` counts the bidders' answers in every round, so the
    caller can charge the auction's message cost.
    """
    staying = sorted(p for p, c in costs.items() if c is not None and c <= start)
    replies = len(costs)            # first round: every picker answers stay / drop
    price, rounds = start, 1
    if not staying:
        return None, start, rounds, replies
    while len(staying) > 1:
        new_price = price - decrement
        nxt = [p for p in staying if costs[p] <= new_price]
        rounds += 1
        replies += len(staying)
        if not nxt:                 # everyone left dropped together: pick one of them
            return rng.choice(staying), price, rounds, replies
        if len(nxt) == 1:           # the last rival just left: the winner keeps the price it quit at
            return nxt[0], price, rounds, replies
        staying, price = nxt, new_price
    return staying[0], price, rounds, replies


def regret(winner_cost, proposals):
    """How much more the chosen picker's true cost is than the cheapest bidder's."""
    return winner_cost - min(p["cost"] for p in proposals)


def solve_winner_determination(bids, value_per_tree, time_limit=1.0):
    """
    Combinatorial auction winner determination with CP-SAT.

    ``bids`` is a list of dicts ``{"picker", "areas": tuple, "trees": int, "bid", "cost"}``.
    Constraints: each area is sold at most once and each picker wins at most one bundle.
    Objective: maximise the auctioneer's surplus, sum(value_per_tree * trees - bid).
    Returns the list of winning bids.
    """
    useful = [b for b in bids if value_per_tree * b["trees"] - b["bid"] > 0]
    if not useful:
        return []
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"b{i}") for i in range(len(useful))]
    for area in {a for b in useful for a in b["areas"]}:
        model.Add(sum(x[i] for i, b in enumerate(useful) if area in b["areas"]) <= 1)
    for picker in {b["picker"] for b in useful}:
        model.Add(sum(x[i] for i, b in enumerate(useful) if b["picker"] == picker) <= 1)
    # CP-SAT needs integer coefficients: work in hundredths of a cost unit
    model.Maximize(sum(int(round(100 * (value_per_tree * b["trees"] - b["bid"]))) * x[i]
                       for i, b in enumerate(useful)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1          # deterministic
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []
    return [b for i, b in enumerate(useful) if solver.Value(x[i])]
