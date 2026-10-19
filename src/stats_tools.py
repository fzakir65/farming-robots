"""Statistics for comparing simulation configurations.

Simulation outputs are rarely normal and the samples are small (one value per seed),
so the comparisons are non-parametric:

* **Mann-Whitney U** test of each configuration against a baseline (two-sided);
* **Holm-Bonferroni** correction, because several configurations are compared
  against the same baseline;
* **Vargha-Delaney A12** effect size: the probability that a random run of
  configuration A beats a random run of B (0.5 = no difference; >= 0.71 or <= 0.29
  is conventionally a "large" effect);
* a **bootstrap 95% confidence interval** for the difference in means.
"""
import numpy as np
import pandas as pd
from scipy import stats


def a12(x, y):
    """Vargha-Delaney A12: P(X > Y) + 0.5 P(X = Y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    greater = (x[:, None] > y[None, :]).sum()
    equal = (x[:, None] == y[None, :]).sum()
    return (greater + 0.5 * equal) / (len(x) * len(y))


def effect_label(a):
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


def bootstrap_diff_ci(x, y, n=5000, seed=0):
    """95% CI of mean(x) - mean(y), by the percentile bootstrap."""
    res = stats.bootstrap((np.asarray(x, float), np.asarray(y, float)),
                          lambda a, b, axis: np.mean(a, axis=axis) - np.mean(b, axis=axis),
                          n_resamples=n, confidence_level=0.95, method="percentile",
                          random_state=np.random.default_rng(seed), vectorized=True)
    return res.confidence_interval.low, res.confidence_interval.high


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def compare(frame, group, metric, baseline):
    """One row per configuration: mean, CI, and the test against ``baseline``."""
    base = frame.loc[frame[group] == baseline, metric].to_numpy()
    rows = []
    for name, sub in frame.groupby(group, sort=False):
        x = sub[metric].to_numpy()
        row = {group: name, "n": len(x), "mean": x.mean(), "std": x.std(ddof=1) if len(x) > 1 else 0.0}
        if name != baseline and len(x) > 1 and len(base) > 1:
            if np.ptp(np.concatenate([x, base])) == 0:
                row["p"] = 1.0
            else:
                row["p"] = stats.mannwhitneyu(x, base, alternative="two-sided").pvalue
            row["A12"] = a12(x, base)
            row["effect"] = effect_label(row["A12"])
            row["diff_lo"], row["diff_hi"] = bootstrap_diff_ci(x, base)
        rows.append(row)
    out = pd.DataFrame(rows)
    if "p" in out:
        mask = out["p"].notna()
        out.loc[mask, "p_holm"] = holm(out.loc[mask, "p"])
        out["significant"] = out["p_holm"] < 0.05
    return out
