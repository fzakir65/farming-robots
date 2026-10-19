"""Experiments for the advanced features, with statistics, sensitivity analysis and tuning.

Usage (from ``src/``; results go to ../results/advanced):
    python advanced_experiments.py compare            # all comparison experiments (parallel)
    python advanced_experiments.py compare --only allocation drones
    python advanced_experiments.py llm --seeds 3      # LLM coordinator (local Ollama; slow)
    python advanced_experiments.py xmpp --seeds 3     # real XMPP network vs ideal bus
    python advanced_experiments.py batch              # Mesa batch_run fleet-design grid
    python advanced_experiments.py sobol              # SALib Sobol sensitivity analysis
    python advanced_experiments.py optuna             # Optuna multi-objective fleet design
    python advanced_experiments.py stats              # statistical tests on saved results
    python advanced_experiments.py all                # everything except llm/xmpp

Each comparison runs every configuration on the same seeds (0..n-1) for 1000 steps,
so differences come from the configuration, not from luck. ``stats_tools.py`` holds
the tests (Mann-Whitney U, Holm correction, Vargha-Delaney A12, bootstrap CIs).
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from farm.constants import (BDI, BOLD, BROADCAST, CAUTIOUS, CBS, CENTRAL_VRP, COMBINATORIAL,  # noqa: E402
                            CONTRACT_NET, DIRECT, ENGLISH, EXTENDED, FICTITIOUS, IDEAL, INDEPENDENT, LLM,
                            LOSSY, NASH, NOVEL, RL, ROAM, RULES, VICKREY, XMPP)
from farm.model import (FarmModel, blocked_steps, energy_per_strawberry, harvest_rate,  # noqa: E402
                        mean_detect_latency, mean_pick_latency, mean_regret, overpayment)
from stats_tools import compare  # noqa: E402

OUT = os.path.join("..", "results", "advanced")
INK, MUTED, GRID, HUE = "#2b2b2b", "#6b6b6b", "#e6e6e3", "#2a78d6"
BASE_HUE = "#8a8a85"

# experiment -> (baseline label, [(label, model kwargs)], KPIs to plot)
EXPERIMENTS = {
    "allocation": ("Contract Net", [
        ("Contract Net", dict(mode=NOVEL, allocation=CONTRACT_NET)),
        ("Vickrey", dict(mode=NOVEL, allocation=VICKREY)),
        ("English", dict(mode=NOVEL, allocation=ENGLISH)),
        ("Combinatorial", dict(mode=NOVEL, allocation=COMBINATORIAL)),
        ("Central VRP", dict(mode=NOVEL, allocation=CENTRAL_VRP)),
        ("Contract Net, 50% strategic", dict(mode=NOVEL, allocation=CONTRACT_NET, strategic_bidders=0.5)),
        ("Vickrey, 50% strategic", dict(mode=NOVEL, allocation=VICKREY, strategic_bidders=0.5)),
        ("Combinatorial, 50% strategic", dict(mode=NOVEL, allocation=COMBINATORIAL, strategic_bidders=0.5)),
    ], ["Harvest rate", "Energy per strawberry", "Messages sent", "Mean regret", "Overpayment"]),
    "brains": ("Rule-based", [
        ("Rule-based", dict(mode=EXTENDED, comm_mode=BROADCAST, picker_brain=RULES)),
        ("BDI, bold", dict(mode=EXTENDED, comm_mode=BROADCAST, picker_brain=BDI, bdi_commitment=BOLD)),
        ("BDI, cautious", dict(mode=EXTENDED, comm_mode=BROADCAST, picker_brain=BDI, bdi_commitment=CAUTIOUS)),
        ("Learned (PPO)", dict(mode=EXTENDED, comm_mode=BROADCAST, picker_brain=RL)),
    ], ["Harvest rate", "Energy per strawberry", "Mean pick latency", "Run seconds"]),
    "paths4": ("Independent", [
        ("Independent", dict(mode=EXTENDED, comm_mode=DIRECT, n_pickers=4, path_planner=INDEPENDENT)),
        ("Windowed CBS", dict(mode=EXTENDED, comm_mode=DIRECT, n_pickers=4, path_planner=CBS)),
    ], ["Harvest rate", "Blocked steps", "Energy per strawberry", "Run seconds"]),
    "paths8": ("Independent", [
        ("Independent", dict(mode=EXTENDED, comm_mode=DIRECT, n_pickers=8, path_planner=INDEPENDENT)),
        ("Windowed CBS", dict(mode=EXTENDED, comm_mode=DIRECT, n_pickers=8, path_planner=CBS)),
    ], ["Harvest rate", "Blocked steps", "Energy per strawberry", "Run seconds"]),
    "drones": ("Random roaming", [
        ("Random roaming", dict(mode=EXTENDED, comm_mode=DIRECT, drone_strategy=ROAM)),
        ("Nash sectors", dict(mode=EXTENDED, comm_mode=DIRECT, drone_strategy=NASH)),
        ("Fictitious play", dict(mode=EXTENDED, comm_mode=DIRECT, drone_strategy=FICTITIOUS)),
    ], ["Mean detection latency", "Harvest rate", "Mean pick latency"]),
    "networks": ("Ideal", [
        ("Ideal", dict(mode=NOVEL, network=IDEAL)),
        ("Lossy 10%", dict(mode=NOVEL, network=LOSSY, drop_prob=0.1)),
        ("Lossy 30%", dict(mode=NOVEL, network=LOSSY, drop_prob=0.3)),
        ("Lossy 50%", dict(mode=NOVEL, network=LOSSY, drop_prob=0.5)),
        ("Extended direct, lossy 30%", dict(mode=EXTENDED, comm_mode=DIRECT, network=LOSSY, drop_prob=0.3)),
    ], ["Harvest rate", "Mean pick latency", "Messages dropped"]),
}

KPI_FUNCS = {
    "Harvest rate": harvest_rate,
    "Energy per strawberry": energy_per_strawberry,
    "Mean pick latency": mean_pick_latency,
    "Mean detection latency": mean_detect_latency,
    "Blocked steps": blocked_steps,
    "Mean regret": mean_regret,
    "Overpayment": overpayment,
}


def run_config(job):
    """Run one configuration for one seed and return its KPIs (used by the worker pool)."""
    experiment, label, kwargs, seed, steps = job
    t0 = time.perf_counter()
    model = FarmModel(seed=seed, max_steps=steps, **kwargs)
    try:
        while model.running:
            model.step()
    finally:
        model.close()
    row = {"Experiment": experiment, "Config": label, "Seed": seed,
           "Run seconds": time.perf_counter() - t0,
           "Delivered": model.delivered, "Wasted": model.wasted, "Messages sent": model.bus.sent,
           "Messages dropped": getattr(model.bus, "dropped", 0),
           "Contracts": model.contracts_awarded, "Payments": model.total_payments}
    row.update({k: f(model) for k, f in KPI_FUNCS.items()})
    if model.paths is not None:
        row.update({"CBS calls": model.paths.calls, "CBS conflicts": model.paths.conflicts,
                    "CBS fallbacks": model.paths.fallbacks})
    if model.llm is not None:
        row.update({"LLM calls": model.llm.calls, "LLM seconds": model.llm.seconds,
                    "LLM invalid": model.llm.invalid, "LLM fallbacks": model.llm_fallbacks})
    if hasattr(model.bus, "mean_latency_ms"):
        row.update({"XMPP latency ms": model.bus.mean_latency_ms, "XMPP copies": model.bus.copies_received})
    if model.picker_brain == BDI:
        row["BDI reconsiderations"] = sum(p.brain.reconsiderations for p in model.pickers)
    if model.picker_brain == RL:
        row["RL trained"] = model.pickers[0].brain.trained
    return row


WORKERS = None   # --workers; default: all cores but two


def pool(processes=None):
    return mp.get_context("spawn").Pool(processes or WORKERS or max(1, (os.cpu_count() or 2) - 2))


# ---------------------------------------------------------------------- #
# Plotting
# ---------------------------------------------------------------------- #
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_comparison(frame, baseline, kpis, title, path):
    """Small multiples: one panel per KPI, mean and 95% CI per configuration (labelled rows)."""
    labels = list(dict.fromkeys(frame["Config"]))
    fig, axes = plt.subplots(1, len(kpis), figsize=(3.6 * len(kpis), 0.5 * len(labels) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kpi in zip(axes, kpis):
        for i, label in enumerate(labels):
            x = frame.loc[frame["Config"] == label, kpi].dropna().to_numpy()
            if len(x) == 0:
                continue
            m = x.mean()
            half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
            colour = BASE_HUE if label == baseline else HUE
            ax.errorbar(m, i, xerr=half, fmt="o", color=colour, ecolor=colour, elinewidth=2, capsize=0,
                        markersize=8)
            ax.annotate(f"{m:.2f}" if abs(m) < 10 else f"{m:.0f}", (m, i), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7, color=INK)
        ax.set_title(kpi, fontsize=10, color=INK, loc="left")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8, color=INK)
        ax.invert_yaxis()
        style(ax)
    fig.suptitle(f"{title} (mean and 95% CI over seeds; grey = baseline)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- #
# Experiments
# ---------------------------------------------------------------------- #
def run_compare(names, seeds, steps):
    jobs = [(name, label, kwargs, seed, steps)
            for name in names for label, kwargs in EXPERIMENTS[name][1] for seed in range(seeds)]
    print(f"{len(jobs)} runs ...", flush=True)
    t0 = time.time()
    with pool() as p:
        rows = p.map(run_config, jobs, chunksize=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    frame = pd.DataFrame(rows)
    path = os.path.join(OUT, "comparisons.csv")
    if os.path.exists(path):                    # keep experiments that were not re-run
        old = pd.read_csv(path)
        frame = pd.concat([old[~old["Experiment"].isin(names)], frame], ignore_index=True)
    frame.to_csv(path, index=False)
    for name in names:
        baseline, _, kpis = EXPERIMENTS[name]
        sub = frame[frame["Experiment"] == name]
        plot_comparison(sub, baseline, kpis, name.capitalize(), os.path.join(OUT, f"{name}.png"))
    run_stats()


def run_sequential(experiment, label, kwargs, seeds, steps):
    rows = []
    for seed in range(seeds):
        row = run_config((experiment, label, kwargs, seed, steps))
        rows.append(row)
        print(f"  {label} seed {seed}: harvest {row['Harvest rate']:.3f} in {row['Run seconds']:.0f}s", flush=True)
    return rows


def run_llm(seeds, steps, model_name):
    rows = run_sequential("llm", f"LLM ({model_name})", dict(mode=NOVEL, allocation=LLM, llm_model=model_name),
                          seeds, steps)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "llm.csv"), index=False)


def run_xmpp(seeds, steps):
    rows = run_sequential("xmpp", "Ideal bus", dict(mode=NOVEL), seeds, steps)
    rows += run_sequential("xmpp", "SPADE / XMPP", dict(mode=NOVEL, network=XMPP), seeds, steps)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "xmpp.csv"), index=False)


def run_batch(steps):
    """Fleet-design grid with Mesa's own batch runner (parallel)."""
    import mesa

    params = {"mode": NOVEL, "n_pickers": [2, 4, 6, 8], "n_drones": [1, 2, 4],
              "picker_capacity": [5, 10, 15], "seed": [0, 1, 2], "max_steps": steps}
    t0 = time.time()
    results = mesa.batch_run(FarmModel, parameters=params, iterations=1, max_steps=steps,
                             number_processes=None, data_collection_period=-1, display_progress=False)
    frame = pd.DataFrame(results)
    keep = ["n_pickers", "n_drones", "picker_capacity", "seed", "Harvest rate", "Delivered", "Wasted",
            "Energy per strawberry", "Mean pick latency", "Mean detection latency"]
    frame = frame[keep]
    frame.to_csv(os.path.join(OUT, "batch_fleet.csv"), index=False)
    print(f"batch_run: {len(frame)} runs in {time.time() - t0:.0f}s", flush=True)

    table = frame.groupby(["n_pickers", "n_drones"])["Harvest rate"].mean().unstack()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(table.values, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
    for (i, j), v in np.ndenumerate(table.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(table.columns)), [str(c) for c in table.columns])
    ax.set_yticks(range(len(table.index)), [str(c) for c in table.index])
    ax.set_xlabel("Drones", color=MUTED)
    ax.set_ylabel("Pickers", color=MUTED)
    ax.set_title("Harvest rate by fleet (mean over capacity and seeds)", loc="left", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "batch_fleet.png"), dpi=130)
    plt.close(fig)


SOBOL_PROBLEM = {
    "num_vars": 6,
    "names": ["n_pickers", "n_drones", "picker_capacity", "initial_ripe_patches", "strategic_bidders", "drop_prob"],
    "bounds": [[2, 10.99], [1, 5.99], [4, 20.99], [2, 20.99], [0, 1], [0, 0.5]],
}


def _sobol_job(job):
    i, values, steps = job
    kw = dict(zip(SOBOL_PROBLEM["names"], values))
    for k in ("n_pickers", "n_drones", "picker_capacity", "initial_ripe_patches"):
        kw[k] = int(kw[k])
    m = FarmModel(mode=NOVEL, network=LOSSY, seed=i, max_steps=steps, **kw)
    while m.running:
        m.step()
    return harvest_rate(m), energy_per_strawberry(m)


def run_sobol(n, steps):
    from SALib.analyze import sobol
    from SALib.sample import sobol as sobol_sample

    X = sobol_sample.sample(SOBOL_PROBLEM, n, calc_second_order=False, seed=0)
    t0 = time.time()
    with pool() as p:
        Y = np.array(p.map(_sobol_job, [(i, x, steps) for i, x in enumerate(X)], chunksize=4))
    print(f"Sobol: {len(X)} runs in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for k, name in enumerate(["Harvest rate", "Energy per strawberry"]):
        res = sobol.analyze(SOBOL_PROBLEM, Y[:, k], calc_second_order=False, seed=0)
        for j, param in enumerate(SOBOL_PROBLEM["names"]):
            rows.append({"Output": name, "Parameter": param, "S1": res["S1"][j], "S1_conf": res["S1_conf"][j],
                         "ST": res["ST"][j], "ST_conf": res["ST_conf"][j]})
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(OUT, "sobol.csv"), index=False)
    pd.DataFrame(np.column_stack([X, Y]), columns=SOBOL_PROBLEM["names"] + ["Harvest rate", "Energy per strawberry"]
                 ).to_csv(os.path.join(OUT, "sobol_samples.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, name in zip(axes, ["Harvest rate", "Energy per strawberry"]):
        sub = frame[frame["Output"] == name].sort_values("ST")
        y = np.arange(len(sub))
        ax.barh(y, sub["ST"], xerr=sub["ST_conf"], color=HUE, height=0.55,
                error_kw={"ecolor": MUTED, "elinewidth": 1})
        ax.scatter(sub["S1"], y, color=INK, marker="|", s=200, zorder=3, label="first-order S1")
        ax.set_yticks(y, sub["Parameter"], fontsize=8)
        ax.set_title(f"{name}: total-order index ST", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sobol.png"), dpi=130)
    plt.close(fig)


FLEET_COST = {"picker": 10.0, "drone": 4.0, "storage_slot": 0.2}


def fleet_cost(n_pickers, n_drones, capacity):
    return (FLEET_COST["picker"] * n_pickers + FLEET_COST["drone"] * n_drones
            + FLEET_COST["storage_slot"] * capacity * n_pickers)


def _optuna_job(job):
    params, seed, steps = job
    m = FarmModel(mode=NOVEL, seed=seed, max_steps=steps, **params)
    while m.running:
        m.step()
    return harvest_rate(m)


def run_optuna(trials, seeds, steps):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(directions=["maximize", "minimize"],
                                sampler=optuna.samplers.NSGAIISampler(seed=0, population_size=16),
                                study_name="fleet-design")
    batch = 16
    t0 = time.time()
    with pool() as p:
        done = 0
        while done < trials:
            asked = [study.ask() for _ in range(min(batch, trials - done))]
            params = []
            for t in asked:
                params.append({
                    "n_pickers": t.suggest_int("n_pickers", 1, 10),
                    "n_drones": t.suggest_int("n_drones", 1, 5),
                    "picker_capacity": t.suggest_int("picker_capacity", 4, 20),
                    "allocation": t.suggest_categorical("allocation", [CONTRACT_NET, VICKREY, COMBINATORIAL]),
                })
            results = p.map(_optuna_job, [(pr, s, steps) for pr in params for s in range(seeds)])
            for k, (t, pr) in enumerate(zip(asked, params)):
                harvest = float(np.mean(results[k * seeds:(k + 1) * seeds]))
                study.tell(t, [harvest, fleet_cost(pr["n_pickers"], pr["n_drones"], pr["picker_capacity"])])
            done += len(asked)
    print(f"Optuna: {trials} trials in {time.time() - t0:.0f}s", flush=True)
    frame = study.trials_dataframe()
    frame = frame.rename(columns={"values_0": "Harvest rate", "values_1": "Fleet cost"})
    pareto = {t.number for t in study.best_trials}
    frame["Pareto"] = frame["number"].isin(pareto)
    frame.to_csv(os.path.join(OUT, "optuna_trials.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    other = frame[~frame["Pareto"]]
    front = frame[frame["Pareto"]].sort_values("Fleet cost")
    ax.scatter(other["Fleet cost"], other["Harvest rate"], color=BASE_HUE, s=22, label="trial")
    ax.plot(front["Fleet cost"], front["Harvest rate"], color=HUE, marker="o", markersize=8, linewidth=2,
            label="Pareto front")
    for _, r in front.iterrows():
        ax.annotate(f"{int(r['params_n_pickers'])}P/{int(r['params_n_drones'])}D/{int(r['params_picker_capacity'])}",
                    (r["Fleet cost"], r["Harvest rate"]), xytext=(4, -10), textcoords="offset points",
                    fontsize=7, color=INK)
    ax.set_xlabel("Fleet cost (units)", color=MUTED)
    ax.set_ylabel("Harvest rate", color=MUTED)
    ax.set_title("Fleet design trade-off (Optuna NSGA-II; labels: pickers/drones/capacity)", loc="left",
                 fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "optuna_pareto.png"), dpi=130)
    plt.close(fig)


def run_stats():
    path = os.path.join(OUT, "comparisons.csv")
    frame = pd.read_csv(path)
    tables = []
    for name, (baseline, _, kpis) in EXPERIMENTS.items():
        sub = frame[frame["Experiment"] == name]
        if sub.empty:
            continue
        for kpi in kpis:
            if kpi not in sub or sub[kpi].isna().all():
                continue
            t = compare(sub, "Config", kpi, baseline)
            t.insert(0, "KPI", kpi)
            t.insert(0, "Experiment", name)
            tables.append(t)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(os.path.join(OUT, "stats.csv"), index=False)
    print(f"statistics for {result['Experiment'].nunique()} experiments -> stats.csv", flush=True)
    return result


def plot_rl_curve():
    path = os.path.join(OUT, "rl_training.csv")
    if not os.path.exists(path):
        return
    curve = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(curve["timesteps"], curve["mean_episode_reward"], color=HUE, linewidth=2)
    ax.set_xlabel("Decisions trained on", color=MUTED)
    ax.set_title("Learned picker: mean episode reward (picked - wasted)", loc="left", fontsize=10, color=INK)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rl_training.png"), dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["compare", "llm", "xmpp", "batch", "sobol", "optuna", "stats", "rl", "all"])
    parser.add_argument("--only", nargs="*", choices=list(EXPERIMENTS))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sobol-n", type=int, default=64)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (lower it if RAM is short)")
    args = parser.parse_args()
    global WORKERS
    WORKERS = args.workers
    os.makedirs(OUT, exist_ok=True)
    started = time.time()
    if args.what in ("compare", "all"):
        run_compare(args.only or list(EXPERIMENTS), args.seeds, args.steps)
    if args.what == "llm":
        run_llm(args.seeds, args.steps, args.llm_model)
    if args.what == "xmpp":
        run_xmpp(args.seeds, args.steps)
    if args.what in ("batch", "all"):
        run_batch(min(args.steps, 800))
    if args.what in ("sobol", "all"):
        run_sobol(args.sobol_n, min(args.steps, 600))
    if args.what in ("optuna", "all"):
        run_optuna(args.trials, 3, min(args.steps, 600))
    if args.what == "stats":
        run_stats()
    if args.what in ("rl", "all"):
        plot_rl_curve()
    with open(os.path.join(OUT, "last_run.json"), "w") as f:
        json.dump({"what": args.what, "seconds": round(time.time() - started)}, f)


if __name__ == "__main__":
    main()
