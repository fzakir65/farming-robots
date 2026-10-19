"""Headless experiments: run each operation mode over many seeds and compare KPIs.

Usage (from ``src/``):
    python experiments.py                 # 10 seeds x 1000 steps, results in ../results
    python experiments.py --seeds 3 --steps 300
    python experiments.py --sweep         # also vary the number of pickers

Outputs:
    results/final_kpis.csv     one row per run (KPIs at the last step)
    results/timeseries.csv     every KPI at every step of every run
    results/sweep_pickers.csv  (with --sweep) harvest rate vs fleet size
    results/*.png              comparison plots
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from farm.constants import BASIC, BROADCAST, DIRECT, EXTENDED, NOVEL  # noqa: E402
from farm.model import FarmModel  # noqa: E402

# label, mode, comm_mode
CONFIGS = [
    ("Basic", BASIC, BROADCAST),
    ("Extended (broadcast)", EXTENDED, BROADCAST),
    ("Extended (direct)", EXTENDED, DIRECT),
    ("Novel (Contract Net)", NOVEL, BROADCAST),
]

# Validated categorical palette (fixed order, one colour per configuration).
# Identity is also carried by labels and line styles, never colour alone.
COLOURS = {
    "Basic": "#2a78d6",
    "Extended (broadcast)": "#eb6834",
    "Extended (direct)": "#1baf7a",
    "Novel (Contract Net)": "#eda100",
}
LINESTYLES = {
    "Basic": "-",
    "Extended (broadcast)": "--",
    "Extended (direct)": "-.",
    "Novel (Contract Net)": ":",
}
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e6e6e3"

KPIS = [
    "Delivered",
    "Wasted",
    "Harvest rate",
    "Energy per strawberry",
    "Mean pick latency",
    "Picker utilisation",
    "Messages sent",
]


def run_once(label, mode, comm_mode, seed, steps, **params):
    model = FarmModel(mode=mode, comm_mode=comm_mode, max_steps=steps, seed=seed, **params)
    while model.running:
        model.step()
    series = model.datacollector.get_model_vars_dataframe().reset_index(names="Step")
    series["Config"] = label
    series["Seed"] = seed
    final = series.iloc[-1].to_dict()
    final["Steps run"] = model.schedule.steps
    final["Contracts awarded"] = model.contracts_awarded
    final["Contracts completed"] = model.contracts_completed
    return final, series


def run_all(seeds, steps, **params):
    finals, series = [], []
    for label, mode, comm in CONFIGS:
        for seed in range(seeds):
            final, ts = run_once(label, mode, comm, seed, steps, **params)
            finals.append(final)
            series.append(ts)
        print(f"  done: {label}")
    return pd.DataFrame(finals), pd.concat(series, ignore_index=True)


def summarise(finals):
    table = finals.groupby("Config", sort=False)[KPIS].agg(["mean", "std"]).round(2)
    return table


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_kpi_bars(finals, path):
    """Small multiples: one panel per KPI, configs on the x axis (labelled, not colour-only)."""
    kpis = ["Harvest rate", "Energy per strawberry", "Mean pick latency", "Messages sent"]
    labels = [c[0] for c in CONFIGS]
    fig, axes = plt.subplots(1, len(kpis), figsize=(16, 4.2))
    for ax, kpi in zip(axes, kpis):
        stats = finals.groupby("Config")[kpi].agg(["mean", "std"]).reindex(labels)
        ax.bar(
            range(len(labels)), stats["mean"], yerr=stats["std"], width=0.6,
            color=[COLOURS[l] for l in labels], edgecolor="white", linewidth=2,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3},
        )
        for i, (v, sd) in enumerate(zip(stats["mean"], stats["std"].fillna(0))):
            ax.annotate(f"{v:.2f}" if v < 10 else f"{v:.0f}", (i, v + sd), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8, color=INK)
        ax.set_title(kpi, fontsize=11, color=INK, loc="left")
        style_axis(ax)
    fig.suptitle("KPIs by operation mode (mean ± std over seeds)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_over_time(series, kpi, path, title):
    """One line per config, direct-labelled at the right end."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mean = series.groupby(["Config", "Step"])[kpi].mean()
    ends = []
    for label, _, _ in CONFIGS:
        if label not in mean.index.get_level_values(0):
            continue
        s = mean.loc[label]
        ax.plot(s.index, s.values, color=COLOURS[label], linestyle=LINESTYLES[label], linewidth=2, label=label)
        ends.append((s.values[-1], s.index[-1], label))
    # direct labels, nudged apart so they never collide
    ends.sort()
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    last_y = None
    for y, x, label in ends:
        y_text = y if last_y is None else max(y, last_y + span * 0.05)
        ax.annotate(label, (x, y), xytext=(x + series["Step"].max() * 0.01, y_text),
                    fontsize=8, color=INK, va="center")
        last_y = y_text
    ax.set_xlim(0, series["Step"].max() * 1.22)
    ax.set_xlabel("Step", color=MUTED)
    ax.set_title(title, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sweep_pickers(seeds, steps, counts=(2, 4, 6, 8)):
    """How does the fleet size change the harvest rate in each ageing-crop mode?"""
    rows = []
    for label, mode, comm in CONFIGS[1:]:
        for n in counts:
            for seed in range(seeds):
                final, _ = run_once(label, mode, comm, seed, steps, n_pickers=n)
                rows.append({"Config": label, "Pickers": n, **{k: final[k] for k in KPIS}})
        print(f"  sweep done: {label}")
    return pd.DataFrame(rows)


def plot_sweep(sweep, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, _, _ in CONFIGS[1:]:
        s = sweep[sweep["Config"] == label].groupby("Pickers")["Harvest rate"].mean()
        ax.plot(s.index, s.values, color=COLOURS[label], linestyle=LINESTYLES[label],
                marker="o", markersize=7, linewidth=2, label=label)
    ax.set_xlabel("Number of picker robots", color=MUTED)
    ax.set_title("Harvest rate vs fleet size", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sweep", action="store_true", help="also vary the number of pickers")
    parser.add_argument("--out", default=os.path.join("..", "results"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Running {len(CONFIGS)} configurations x {args.seeds} seeds x up to {args.steps} steps")
    finals, series = run_all(args.seeds, args.steps)
    finals.to_csv(os.path.join(args.out, "final_kpis.csv"), index=False)
    series.to_csv(os.path.join(args.out, "timeseries.csv"), index=False)

    print("\nSummary (mean / std):")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summarise(finals))

    plot_kpi_bars(finals, os.path.join(args.out, "kpi_comparison.png"))
    ageing = series[series["Config"] != "Basic"]
    plot_over_time(ageing, "Delivered", os.path.join(args.out, "delivered_over_time.png"),
                   "Strawberries delivered to base (ageing-crop modes)")
    plot_over_time(ageing, "Wasted", os.path.join(args.out, "wasted_over_time.png"),
                   "Strawberries wasted by rotting (ageing-crop modes)")

    if args.sweep:
        sweep = sweep_pickers(max(3, args.seeds // 2), args.steps)
        sweep.to_csv(os.path.join(args.out, "sweep_pickers.csv"), index=False)
        plot_sweep(sweep, os.path.join(args.out, "sweep_pickers.png"))
    print(f"\nResults written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
