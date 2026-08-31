"""Publication-ready Matplotlib figure export for PIR-Framework evaluation.

Styling follows common IEEE/PMLR conference conventions: serif fonts,
vector output (PDF primary, EPS also emitted for LaTeX \\includegraphics
compatibility with older toolchains), high DPI for any rasterized
elements, minimal chart junk (no top/right spines, light gridlines), and
explicit legends on every multi-series figure. Every `plot_*` function
still also writes a PNG for quick inline viewing/preview.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless rendering: safe for scripts/CI, no display needed
import matplotlib.pyplot as plt

IEEE_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,   # embed TrueType (editable text) rather than Type-3, in PDF
    "ps.fonttype": 42,    # same for EPS
    "svg.fonttype": "none",
}
plt.rcParams.update(IEEE_RC)

# A colorblind-safe, print-friendly qualitative palette (Okabe-Ito), used
# consistently across all multi-series figures so the same policy always
# gets the same color.
POLICY_COLORS = {
    "Fixed-Rate": "#E69F00",
    "Always-Local": "#56B4E9",
    "Always-Edge": "#009E73",
    "Heuristic-Rule": "#CC79A7",
    "PIR-Contextual-Bandit (LinUCB)": "#D55E00",
    "PIR-Contextual-Bandit (Thompson)": "#0072B2",
}
DEFAULT_CYCLE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00", "#0072B2", "#999999"]


def _policy_color(name: str, fallback_idx: int = 0) -> str:
    if name in POLICY_COLORS:
        return POLICY_COLORS[name]
    return DEFAULT_CYCLE[fallback_idx % len(DEFAULT_CYCLE)]


def _save(fig: plt.Figure, out_path: str | Path, formats: tuple[str, ...] = ("png", "pdf", "eps")) -> dict[str, Path]:
    """Save a figure in every requested vector/raster format. Returns a
    dict of format -> path. EPS does not support alpha transparency, so
    figures intended for EPS export should avoid relying on it for
    legibility (we keep alpha use to light gridlines/CI shading only,
    which still renders acceptably as EPS approximates alpha via
    stippling/solid fallback in most renderers)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = {}
    stem = out_path.with_suffix("")
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        try:
            fig.savefig(p, bbox_inches="tight")
            paths[fmt] = p
        except Exception as e:  # pragma: no cover - defensive; e.g. missing eps backend
            print(f"    [plotting] warning: failed to save {fmt} for {out_path.name}: {e}")
    plt.close(fig)
    return paths


def _legend_below(ax: plt.Axes, ncol: int = 3) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=ncol, fontsize=8, frameon=False)


def _group_coincident_points(
    df: pd.DataFrame, x_col: str, y_col: str, rel_tol: float = 1e-6,
) -> list[tuple[float, float, list[str]]]:
    """Group rows whose (x, y) coordinates coincide (within a small
    relative tolerance -- e.g. two placement-pinned baselines that share
    identical energy/accuracy because placement doesn't affect decode
    probability) so the caller can draw one shared marker/label instead of
    fully overlapping text. Returns a list of (x, y, [policy_names])."""
    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    names = df["policy"].tolist()
    x_scale = max(np.ptp(x), 1e-9)
    y_scale = max(np.ptp(y), 1e-9)

    groups: list[tuple[float, float, list[str]]] = []
    used = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if used[i]:
            continue
        close = (np.abs(x - x[i]) / x_scale < rel_tol) & (np.abs(y - y[i]) / y_scale < rel_tol) & (~used)
        idx = np.where(close)[0]
        used[idx] = True
        groups.append((float(np.mean(x[idx])), float(np.mean(y[idx])), [names[j] for j in idx]))
    return groups


def _place_labels_no_overlap(
    ax: plt.Axes, points: list[tuple[float, float, str]], base_offset: tuple[float, float] = (7, 5),
    fontsize: float = 7.5,
) -> None:
    """Lightweight label-declutter: places each (x, y, text) annotation at
    `base_offset`, then nudges any label whose axes-fraction position falls
    within a minimum distance of an already-placed label further along a
    small set of candidate offsets, until it clears every prior label or
    the candidates are exhausted (falls back to the largest offset tried).
    This avoids a hard dependency on an external label-placement package
    while still resolving the common case of 2-4 nearby/coincident points.
    """
    placed_axes_xy: list[tuple[float, float]] = []
    candidate_offsets = [base_offset, (base_offset[0], base_offset[1] + 14),
                          (base_offset[0], base_offset[1] - 14), (base_offset[0] + 34, base_offset[1]),
                          (base_offset[0], base_offset[1] + 28), (base_offset[0], base_offset[1] - 28)]
    min_axes_dist = 0.055

    for x, y, text in points:
        data_to_axes = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
        chosen = candidate_offsets[0]
        for off in candidate_offsets:
            # Approximate the annotation's axes-fraction position under this
            # pixel offset (offset is in points; ~1 point ~ 1/72 inch).
            fig_dpi = ax.figure.dpi
            dx_axes = (off[0] / 72.0 * fig_dpi) / ax.figure.bbox.width
            dy_axes = (off[1] / 72.0 * fig_dpi) / ax.figure.bbox.height
            candidate_axes_xy = (data_to_axes[0] + dx_axes, data_to_axes[1] + dy_axes)
            if all(
                (candidate_axes_xy[0] - px) ** 2 + (candidate_axes_xy[1] - py) ** 2 >= min_axes_dist ** 2
                for px, py in placed_axes_xy
            ):
                chosen = off
                placed_axes_xy.append(candidate_axes_xy)
                break
        else:
            fig_dpi = ax.figure.dpi
            off = candidate_offsets[-1]
            dx_axes = (off[0] / 72.0 * fig_dpi) / ax.figure.bbox.width
            dy_axes = (off[1] / 72.0 * fig_dpi) / ax.figure.bbox.height
            placed_axes_xy.append((data_to_axes[0] + dx_axes, data_to_axes[1] + dy_axes))
            chosen = off

        ax.annotate(text, (x, y), textcoords="offset points", xytext=chosen,
                    ha="left" if chosen[0] >= 0 else "right", fontsize=fontsize, zorder=4)


def plot_policy_comparison_bars(
    df: pd.DataFrame, metric: str, out_path: str | Path,
    ylabel: str | None = None, title: str | None = None, lower_is_better: bool = False,
    ci_col: str | None = None,
) -> dict[str, Path]:
    """Bar chart of `metric` (one bar per policy), sorted best-to-worst.
    If `ci_col` names a column holding a half-width (e.g. 95% CI), error
    bars are drawn from it."""
    d = df.sort_values(metric, ascending=lower_is_better)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    colors = [_policy_color(p, i) for i, p in enumerate(d["policy"])]
    yerr = d[ci_col].to_numpy() if ci_col and ci_col in d.columns else None
    ax.bar(d["policy"], d[metric], color=colors, yerr=yerr, capsize=3,
           error_kw={"linewidth": 1.0, "ecolor": "#333333"})
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, out_path)


def plot_energy_latency_tradeoff(
    df: pd.DataFrame, out_path: str | Path, deadline_ms: float = 150.0,
) -> dict[str, Path]:
    """Scatter of mean energy vs. mean latency per policy, marker size ~ PDR.
    Points below/left of the deadline line and toward the origin are more
    Pareto-favorable (lower energy, lower latency); the Pareto-optimal
    subset (not dominated by any other policy on both axes) is outlined."""
    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    is_pareto = _pareto_front(df["mean_energy_j"].to_numpy(), df["mean_latency_ms"].to_numpy())
    pareto_by_policy = dict(zip(df["policy"], is_pareto))

    groups = _group_coincident_points(df, "mean_energy_j", "mean_latency_ms")
    label_points = []
    for gx, gy, names in groups:
        any_pareto = any(pareto_by_policy[n] for n in names)
        row0 = df[df["policy"] == names[0]].iloc[0]
        color = _policy_color(names[0], list(df["policy"]).index(names[0]))
        size = 120 + 500 * row0["pdr"]
        marker = "*" if any_pareto else "o"
        edge = "black" if any_pareto else "#666666"
        lw = 1.6 if any_pareto else 0.8
        ax.scatter(gx, gy, s=size * (1.6 if any_pareto else 1.0), color=color, marker=marker,
                   edgecolors=edge, linewidths=lw, zorder=3)
        label_points.append((gx, gy, " / ".join(names)))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter([], [], color=_policy_color(row["policy"], i), label=row["policy"])  # legend-only proxy
    _place_labels_no_overlap(ax, label_points)

    ax.axhline(deadline_ms, color="#B22222", linestyle="--", linewidth=1.2, zorder=1,
               label=f"{deadline_ms:.0f} ms deadline")
    ax.set_xlabel("Mean energy per window (J)")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Energy-latency tradeoff (marker size $\\propto$ PDR; $\\star$ = Pareto-optimal)")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="best", fontsize=7.5, framealpha=0.9)
    return _save(fig, out_path)


def _pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask over points that are not dominated (both x and y
    lower-or-equal, at least one strictly lower) by any other point --
    i.e. the minimization Pareto frontier for (x, y)."""
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i]):
                dominated[i] = True
                break
    return ~dominated


def plot_pareto_frontier(
    df: pd.DataFrame, out_path: str | Path,
    x_col: str = "mean_power_mw", y_col: str = "hr_mae_bpm",
    x_label: str = "Mean power (mW)", y_label: str = "Heart-rate MAE (bpm)",
    power_cap_mw: float | None = 100.0,
) -> dict[str, Path]:
    """
    Explicit Pareto-frontier plot for the two objectives most directly named
    in the PIR-Controller's constraints: average power draw (mW, capped at
    `power_cap_mw`) on the x-axis vs. delivered clinical accuracy (heart-rate
    MAE, bpm -- lower is better) on the y-axis. Both objectives are
    "lower is better", so the frontier is the lower-left envelope; policies
    on the frontier are starred and connected with a dashed guide line.
    """
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    is_pareto = _pareto_front(x, y)
    pareto_by_policy = dict(zip(df["policy"], is_pareto))

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    groups = _group_coincident_points(df, x_col, y_col)
    label_points = []
    for gx, gy, names in groups:
        any_pareto = any(pareto_by_policy[n] for n in names)
        color = _policy_color(names[0], list(df["policy"]).index(names[0]))
        marker = "*" if any_pareto else "o"
        size = 260 if any_pareto else 110
        ax.scatter(gx, gy, s=size, color=color, marker=marker,
                   edgecolors="black" if any_pareto else "#666666",
                   linewidths=1.4 if any_pareto else 0.8, zorder=3)
        label_points.append((gx, gy, " / ".join(names)))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter([], [], color=_policy_color(row["policy"], i), label=row["policy"])  # legend-only proxy
    _place_labels_no_overlap(ax, label_points)

    if np.any(is_pareto):
        pf_x, pf_y = x[is_pareto], y[is_pareto]
        order = np.argsort(pf_x)
        ax.plot(pf_x[order], pf_y[order], linestyle="--", color="#444444", linewidth=1.0, zorder=2,
                label="Pareto frontier")

    if power_cap_mw is not None and x_col == "mean_power_mw":
        ax.axvline(power_cap_mw, color="#B22222", linestyle=":", linewidth=1.2, label=f"{power_cap_mw:.0f} mW power cap")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Pareto frontier: accuracy vs. power")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="best", fontsize=7.5, framealpha=0.9)
    return _save(fig, out_path)


def plot_regret_curve(
    cumulative_reward_by_policy: dict[str, np.ndarray], out_path: str | Path,
    oracle_reward: np.ndarray | None = None,
) -> dict[str, Path]:
    """Cumulative-reward (and, if an oracle reward trace is supplied,
    cumulative-regret) curves across bandit training rounds. Single-seed
    version (no shading); see `plot_regret_curve_with_ci` for the
    multi-seed, confidence-banded version used in the main evaluation."""
    fig, axes = plt.subplots(1, 2 if oracle_reward is not None else 1,
                              figsize=(10.5, 3.8) if oracle_reward is not None else (5.8, 3.8))
    ax0 = axes[0] if oracle_reward is not None else axes
    for i, (name, trace) in enumerate(cumulative_reward_by_policy.items()):
        ax0.plot(np.cumsum(trace), label=name, color=_policy_color(name, i))
    ax0.set_xlabel("Round")
    ax0.set_ylabel("Cumulative reward")
    ax0.set_title("Cumulative reward")
    _legend_below(ax0, ncol=2)

    if oracle_reward is not None:
        ax1 = axes[1]
        oracle_cum = np.cumsum(oracle_reward)
        for i, (name, trace) in enumerate(cumulative_reward_by_policy.items()):
            regret = oracle_cum - np.cumsum(trace)
            ax1.plot(regret, label=name, color=_policy_color(name, i))
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Cumulative regret")
        ax1.set_title("Cumulative regret vs. oracle")
        _legend_below(ax1, ncol=2)
    return _save(fig, out_path)


def plot_regret_curve_with_ci(
    traces_by_policy: dict[str, np.ndarray],  # policy -> array (n_seeds, n_rounds)
    out_path: str | Path,
    ylabel: str = "Cumulative reward",
    title: str = "Bandit convergence (mean $\\pm$ 95% CI over seeds)",
    downsample_to: int = 400,
) -> dict[str, Path]:
    """
    Multi-seed cumulative-reward convergence plot with a shaded 95%
    confidence interval (mean +/- 1.96 * SEM across seeds) around each
    policy's curve -- the standard way bandit-convergence figures are
    presented in ML venues (e.g. PMLR papers).

    `traces_by_policy[name]` must be a (n_seeds, n_rounds) array of
    *per-round* (not yet cumulative-summed) rewards; this function performs
    the cumulative sum per-seed before aggregating, so seed-to-seed noise
    in the *level* of the curve is preserved in the band rather than
    averaged away first.
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, (name, traces) in enumerate(traces_by_policy.items()):
        traces = np.asarray(traces, dtype=np.float64)
        cum = np.cumsum(traces, axis=1)  # (n_seeds, n_rounds)
        n_seeds, n_rounds = cum.shape

        step = max(1, n_rounds // downsample_to)
        idx = np.arange(0, n_rounds, step)
        cum_ds = cum[:, idx]

        mean = cum_ds.mean(axis=0)
        sem = cum_ds.std(axis=0, ddof=1) / np.sqrt(max(n_seeds, 1)) if n_seeds > 1 else np.zeros_like(mean)
        ci95 = 1.96 * sem

        color = _policy_color(name, i)
        ax.plot(idx, mean, color=color, label=f"{name} (n={n_seeds} seeds)")
        if n_seeds > 1:
            ax.fill_between(idx, mean - ci95, mean + ci95, color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend_below(ax, ncol=2)
    return _save(fig, out_path)


def plot_pdr_vs_snr(
    snr_db: np.ndarray, pdr_success: np.ndarray, out_path: str | Path, n_bins: int = 15,
    label: str = "Empirical decode success rate",
) -> dict[str, Path]:
    """Binned scatter/line of empirical decode success rate vs. SNR (dB),
    with a Wilson-score 95% CI band per bin."""
    bins = np.linspace(snr_db.min(), snr_db.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(snr_db, bins) - 1, 0, n_bins - 1)
    centers, rates, los, his = [], [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(np.sum(mask))
        if n == 0:
            continue
        k = int(np.sum(pdr_success[mask]))
        p_hat = k / n
        lo, hi = _wilson_ci(k, n)
        centers.append(0.5 * (bins[b] + bins[b + 1]))
        rates.append(p_hat)
        los.append(lo)
        his.append(hi)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    centers, rates, los, his = map(np.array, (centers, rates, los, his))
    ax.fill_between(centers, los, his, color="#0072B2", alpha=0.18, linewidth=0, label="95% Wilson CI")
    ax.plot(centers, rates, marker="o", markersize=4, color="#0072B2", label=label)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Decode success rate")
    ax.set_title("Decode success vs. channel SNR")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out_path)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion
    k/n -- more reliable than the normal approximation at small n or
    p_hat near 0/1, both of which occur here (near-clean and near-total-
    erasure SNR bins)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def plot_arm_allocation(
    arm_names: list[str], pull_counts: np.ndarray, out_path: str | Path,
    title: str = "Learned arm allocation",
) -> dict[str, Path]:
    """Bar chart of how often the contextual bandit selected each arm."""
    order = np.argsort(pull_counts)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = plt.cm.cividis(np.linspace(0.15, 0.85, len(arm_names)))
    ax.bar(np.array(arm_names)[order], pull_counts[order], color=colors[order])
    ax.set_ylabel("Number of selections")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, out_path)
