"""
Comprehensive evaluation pipeline: compares five control policies for the
joint (RLNC redundancy, generation size, compute-placement) decision, on a
shared stream of physiological windows spanning six CardioFusion-AI-style
degradation regimes, over a Gilbert-Elliott wireless channel.

Policies compared
------------------
  Fixed-Rate            Static arm (K=32, R=8, local), never adapts.
  Always-Local           Fixed redundancy, placement pinned to local compute.
  Always-Edge             Fixed redundancy, placement pinned to edge offload.
  Heuristic-Rule          Hand-tuned if/else thresholds on battery / erasure
                          estimate (a genuine, non-trivial rule-based
                          baseline -- included explicitly as the "obvious"
                          alternative the contextual bandit is compared
                          against, not a strawman).
  PIR-Contextual-Bandit   LinUCB (default) or Thompson Sampling, learned
                          online from the reward signal defined in
                          `pir_controller.py`.

Metrics reported (per policy x regime, and overall): Heart-Rate MAE (bpm),
clinical event-detection F1, Packet Delivery Ratio, energy per window (J),
average power (mW), latency (ms), deadline-success rate, power-cap-success
rate, and mean reward.

Produces: a per-(policy, regime) CSV, an overall-per-policy CSV, a
publication-ready LaTeX table (.tex), and a set of Matplotlib figures
(policy comparison bars, energy-latency tradeoff, cumulative reward/regret,
PDR vs. SNR, and learned arm allocation for the bandit).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data_pipeline import PhysiologicalDataset, SyntheticECGPPGGenerator, build_dataset
from .state_space import StateSpaceEngine, MedicalState
from .pir_controller import (
    Arm, LinUCBController, ThompsonSamplingController, RewardWeights,
    compute_reward, default_arm_set,
)
from .utils.metrics import EvaluationMetrics, compute_window_metrics, aggregate_metrics, overall_metrics
from .utils.latex_export import dataframe_to_latex_table
from .utils.plotting import (
    plot_policy_comparison_bars, plot_energy_latency_tradeoff, plot_pareto_frontier,
    plot_regret_curve, plot_regret_curve_with_ci, plot_pdr_vs_snr, plot_arm_allocation,
)


# --------------------------------------------------------------------------
# Policy definitions
# --------------------------------------------------------------------------


class BasePolicy:
    """Interface: `select(contexts, dataset_slice) -> arm_indices`."""

    name: str

    def select(self, contexts: np.ndarray, extra: dict) -> np.ndarray:
        raise NotImplementedError

    def update(self, contexts: np.ndarray, arm_indices: np.ndarray, rewards: np.ndarray) -> None:
        pass  # non-adaptive policies ignore feedback


class FixedArmPolicy(BasePolicy):
    def __init__(self, name: str, arms: list[Arm], fixed_arm_index: int):
        self.name = name
        self.arms = arms
        self.fixed_arm_index = fixed_arm_index

    def select(self, contexts: np.ndarray, extra: dict) -> np.ndarray:
        return np.full(contexts.shape[0], self.fixed_arm_index, dtype=np.int64)


class PlacementPinnedPolicy(BasePolicy):
    """Fixes placement (local or edge) but still fixes a single redundancy
    level -- used for the Always-Local / Always-Edge baselines."""

    def __init__(self, name: str, arms: list[Arm], placement_is_edge: bool, r_value: int):
        self.name = name
        self.arms = arms
        candidates = [i for i, a in enumerate(arms) if a.placement_is_edge == placement_is_edge and a.R == r_value]
        if not candidates:
            raise ValueError(f"No arm matches placement={placement_is_edge}, R={r_value}")
        self.arm_index = candidates[0]

    def select(self, contexts: np.ndarray, extra: dict) -> np.ndarray:
        return np.full(contexts.shape[0], self.arm_index, dtype=np.int64)


class HeuristicRulePolicy(BasePolicy):
    """
    Genuine hand-tuned if/else rule baseline (deliberately *not* trivial):

      1. If battery is critically low (<15%), pick the lowest-energy arm
         (minimum R, local placement) to conserve power, regardless of
         channel quality.
      2. Else, estimate the required redundancy from the EWMA erasure-
         probability estimate by picking the smallest R in the arm set
         whose closed-form decode probability at that estimated erasure
         rate clears a fixed target (0.99).
      3. Placement: local if battery < 50% (avoid RTT/offload energy while
         conserving power), else edge (assume edge compute is cheaper once
         power headroom exists).

    This uses the *same* physical quantities (battery fraction, erasure
    estimate) available to the bandit's context, so the comparison isolates
    "hand-written thresholds" vs. "learned linear policy" rather than
    "less information" vs. "more information".
    """

    def __init__(self, name: str, arms: list[Arm], target_decode_prob: float = 0.99,
                 battery_critical: float = 0.15, battery_mid: float = 0.5):
        self.name = name
        self.arms = arms
        self.target_decode_prob = target_decode_prob
        self.battery_critical = battery_critical
        self.battery_mid = battery_mid
        self._sorted_R_local = sorted({a.R for a in arms})

    def select(self, contexts: np.ndarray, extra: dict) -> np.ndarray:
        from .coding.rlnc import decode_probability

        battery_frac = contexts[:, 8]
        p_erasure_est = contexts[:, 6]
        n = contexts.shape[0]
        out = np.empty(n, dtype=np.int64)

        min_r_local_idx = min(
            (i for i, a in enumerate(self.arms) if not a.placement_is_edge),
            key=lambda i: self.arms[i].R,
        )

        for i in range(n):
            if battery_frac[i] < self.battery_critical:
                out[i] = min_r_local_idx
                continue

            chosen_R = self._sorted_R_local[-1]
            for R in self._sorted_R_local:
                K = self.arms[0].K
                p_dec = decode_probability(K, R, float(p_erasure_est[i]))
                if p_dec >= self.target_decode_prob:
                    chosen_R = R
                    break

            placement_is_edge = battery_frac[i] >= self.battery_mid
            candidates = [
                j for j, a in enumerate(self.arms)
                if a.R == chosen_R and a.placement_is_edge == placement_is_edge
            ]
            out[i] = candidates[0] if candidates else 0
        return out


class BanditPolicy(BasePolicy):
    def __init__(self, name: str, controller):
        self.name = name
        self.controller = controller

    def select(self, contexts: np.ndarray, extra: dict) -> np.ndarray:
        return self.controller.select(contexts)

    def update(self, contexts: np.ndarray, arm_indices: np.ndarray, rewards: np.ndarray) -> None:
        self.controller.update(contexts, arm_indices, rewards)


# --------------------------------------------------------------------------
# HR fusion (mirrors CardioFusion-AI's signal-quality-weighted combination)
# --------------------------------------------------------------------------


def slice_medical(medical: MedicalState, idx: int) -> MedicalState:
    """Build a length-1 `MedicalState` for window `idx` of a full-dataset
    `MedicalState`, used to drive the single-environment sequential
    simulation loop in `simulate_policy`."""
    sl = slice(idx, idx + 1)
    return MedicalState(
        sqi_ecg=medical.sqi_ecg[sl], sqi_ppg=medical.sqi_ppg[sl],
        modality_present_ecg=medical.modality_present_ecg[sl],
        modality_present_ppg=medical.modality_present_ppg[sl],
        hr_ecg=medical.hr_ecg[sl], hr_ppg=medical.hr_ppg[sl],
    )


def fuse_hr_estimate(ds: PhysiologicalDataset) -> np.ndarray:
    """Signal-quality-weighted fusion of the per-modality heart-rate
    estimates: hr_fused = (sqi_ecg*hr_ecg + sqi_ppg*hr_ppg) / (sqi_ecg + sqi_ppg),
    falling back to whichever modality is present if the other is fully
    missing, and to the population mean heart rate if both are missing
    (a conservative, uninformative prior)."""
    w_ecg = np.where(ds.present_ecg, ds.sqi_ecg, 0.0)
    w_ppg = np.where(ds.present_ppg, ds.sqi_ppg, 0.0)
    weight_sum = w_ecg + w_ppg
    safe_denom = np.where(weight_sum > 1e-9, weight_sum, 1.0)
    fused = (w_ecg * ds.hr_est_ecg + w_ppg * ds.hr_est_ppg) / safe_denom
    both_missing = weight_sum <= 1e-9
    fused = np.where(both_missing, np.mean(ds.hr_true_bpm), fused)
    return fused


# --------------------------------------------------------------------------
# Simulation driver
# --------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    n_per_regime: int = 200
    battery_capacity_j: float = 200.0
    reward_weights: RewardWeights = None
    bandit_type: str = "linucb"   # "linucb" | "thompson"
    linucb_alpha: float = 0.3
    ridge_lambda: float = 1.0
    thompson_noise_variance: float = 0.05
    seed: int = 0
    outdir: str = "results"

    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = RewardWeights()


def simulate_policy(
    policy: BasePolicy,
    order: np.ndarray,
    medical: MedicalState,
    hr_fused: np.ndarray,
    arms: list[Arm],
    config: "SimulationConfig",
    channel_seed: int,
    shared_decode_draws: np.ndarray,
    population_mean_hr: float,
) -> dict:
    """
    Runs one policy as a *sequential single-device monitoring session*:
    windows arrive one at a time in `order`, the device's Gilbert-Elliott
    channel state and battery evolve continuously across the whole session,
    and (for adaptive policies) the controller is updated after every
    window -- giving `len(order)` genuine online-learning rounds rather
    than a single synchronous batch decision.

    Returns a dict of length-n_total arrays (indexed by the *original*
    window index, not arrival order) plus the final battery fraction.
    """
    n_total = len(order)
    engine = StateSpaceEngine(
        n_envs=1, battery_capacity_j=config.battery_capacity_j,
        seed=channel_seed,  # identical across policies: same channel-state trajectory
    )

    decode_success = np.zeros(n_total, dtype=bool)
    energy_j = np.zeros(n_total)
    power_mw = np.zeros(n_total)
    latency_ms = np.zeros(n_total)
    reward = np.zeros(n_total)
    piv_arr = np.zeros(n_total)
    u_pir_arr = np.zeros(n_total)
    hr_delivered = np.zeros(n_total)
    snr_db = np.zeros(n_total)
    arm_idx_arr = np.zeros(n_total, dtype=np.int64)

    last_hr = population_mean_hr

    for t, idx in enumerate(order):
        med_i = slice_medical(medical, idx)
        ctx = engine.observe_context(med_i)  # (1, d)

        arm_choice = policy.select(ctx, extra={"t": t})
        a = int(arm_choice[0])
        arm_idx_arr[idx] = a
        arm = arms[a]

        step_result = engine.act(
            med_i, K=np.array([arm.K]), R=np.array([arm.R]),
            placement_is_edge=np.array([arm.placement_is_edge]),
        )

        p_dec = float(step_result.p_decode[0])
        success = bool(shared_decode_draws[idx] < p_dec)
        decode_success[idx] = success
        delivered = hr_fused[idx] if success else last_hr
        if success:
            last_hr = hr_fused[idx]
        hr_delivered[idx] = delivered

        energy_j[idx] = step_result.energy_j[0]
        power_mw[idx] = step_result.power_mw[0]
        latency_ms[idx] = step_result.latency_ms[0]
        piv_arr[idx] = med_i.piv[0]
        u_pir_arr[idx] = med_i.piv[0] * p_dec
        snr_db[idx] = step_result.wireless.snr_db[0]

        r, _ = compute_reward(
            piv=med_i.piv, p_decode=step_result.p_decode, energy_j=step_result.energy_j,
            latency_ms=step_result.latency_ms, power_mw=step_result.power_mw, weights=config.reward_weights,
        )
        reward[idx] = float(r[0])

        policy.update(ctx, arm_choice, r)

    return {
        "decode_success": decode_success, "hr_delivered": hr_delivered,
        "energy_j": energy_j, "power_mw": power_mw, "latency_ms": latency_ms,
        "reward": reward, "piv": piv_arr, "u_pir": u_pir_arr, "snr_db": snr_db,
        "arm_idx": arm_idx_arr, "final_battery_frac": float(engine.energy_state.fraction_remaining()[0]),
    }


def run_single_seed(config: SimulationConfig, verbose: bool = True) -> dict:
    """
    Runs the full five-policy comparison for one seed and returns every
    intermediate artifact (dataset, per-window logs, per-(policy,regime)
    and per-policy-overall metrics DataFrames, arrival-order reward traces,
    and arm-pull counts) without writing anything to disk -- `run_evaluation`
    and `run_multi_seed_evaluation` both build on this and handle
    file/plot export themselves.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"[1/5] (seed={config.seed}) Generating physiologically-grounded synthetic dataset ...")
    gen = SyntheticECGPPGGenerator(seed=config.seed)
    ds = build_dataset(n_per_regime=config.n_per_regime, generator=gen, seed=config.seed)
    n_total = len(ds.regime)
    hr_fused = fuse_hr_estimate(ds)
    log(f"    -> {n_total} windows across {len(np.unique(ds.regime))} regimes")

    log("[2/5] Building joint medical state (SQI + PIV) for every window ...")
    medical = MedicalState(
        sqi_ecg=ds.sqi_ecg, sqi_ppg=ds.sqi_ppg,
        modality_present_ecg=ds.present_ecg, modality_present_ppg=ds.present_ppg,
        hr_ecg=ds.hr_est_ecg, hr_ppg=ds.hr_est_ppg,
    )

    arms = default_arm_set()
    context_dim = len(StateSpaceEngine.context_feature_names())
    fixed_arm_index = next(i for i, a in enumerate(arms) if a.R == 8 and not a.placement_is_edge)

    log("[3/5] Instantiating policies (Fixed-Rate, Always-Local, Always-Edge, "
        "Heuristic-Rule, PIR-Contextual-Bandit) ...")
    if config.bandit_type == "linucb":
        controller = LinUCBController(arms, context_dim=context_dim, alpha=config.linucb_alpha,
                                       ridge_lambda=config.ridge_lambda)
        bandit_label = "PIR-Contextual-Bandit (LinUCB)"
    elif config.bandit_type == "thompson":
        controller = ThompsonSamplingController(arms, context_dim=context_dim,
                                                  noise_variance=config.thompson_noise_variance,
                                                  ridge_lambda=config.ridge_lambda, seed=config.seed + 2)
        bandit_label = "PIR-Contextual-Bandit (Thompson)"
    else:
        raise ValueError(f"Unknown bandit_type: {config.bandit_type}")

    policies: list[BasePolicy] = [
        FixedArmPolicy("Fixed-Rate", arms, fixed_arm_index),
        PlacementPinnedPolicy("Always-Local", arms, placement_is_edge=False, r_value=20),
        PlacementPinnedPolicy("Always-Edge", arms, placement_is_edge=True, r_value=20),
        HeuristicRulePolicy("Heuristic-Rule", arms),
        BanditPolicy(bandit_label, controller),
    ]

    log(f"[4/5] Running {len(policies)} policies as sequential single-device monitoring "
        f"sessions over {n_total} windows each (shared window order and shared channel "
        f"random-number stream across policies, for a fair, low-variance comparison) ...")

    # Shared arrival order and a shared decode-outcome random stream across
    # all policies (the "common random numbers" variance-reduction
    # technique): every policy faces the same sequence of windows and the
    # same uniform draws for the erasure-vs-decode coin flip, so observed
    # differences reflect policy quality rather than lucky/unlucky channel
    # draws. Each policy still gets its own Gilbert-Elliott *channel state*
    # trajectory (seeded identically across policies) and its own battery,
    # since those evolve as a consequence of the policy's own actions.
    order = np.random.default_rng(config.seed + 7).permutation(n_total)
    decode_rng = np.random.default_rng(config.seed + 13)
    shared_decode_draws = decode_rng.random(n_total)
    population_mean_hr = float(np.mean(ds.hr_true_bpm))
    channel_seed = config.seed + 1000

    all_records: list[EvaluationMetrics] = []
    per_window_logs: dict[str, pd.DataFrame] = {}
    reward_traces_arrival_order: dict[str, np.ndarray] = {}
    arm_pull_counts: dict[str, np.ndarray] = {}

    for policy in policies:
        result = simulate_policy(
            policy=policy, order=order, medical=medical, hr_fused=hr_fused,
            arms=arms, config=config, channel_seed=channel_seed,
            shared_decode_draws=shared_decode_draws, population_mean_hr=population_mean_hr,
        )
        decode_success = result["decode_success"]
        hr_delivered = result["hr_delivered"]
        energy_j = result["energy_j"]
        power_mw = result["power_mw"]
        latency_ms = result["latency_ms"]
        reward = result["reward"]
        piv_arr = result["piv"]
        u_pir_arr = result["u_pir"]
        chosen_arm_idx = result["arm_idx"]
        snr_db = result["snr_db"]

        reward_traces_arrival_order[policy.name] = reward[order]
        if hasattr(policy, "controller"):
            arm_pull_counts[policy.name] = np.array(
                [np.sum(chosen_arm_idx == i) for i in range(len(arms))], dtype=np.float64,
            )

        df_log = pd.DataFrame({
            "regime": ds.regime, "hr_true": ds.hr_true_bpm, "hr_delivered": hr_delivered,
            "decode_success": decode_success, "energy_j": energy_j, "power_mw": power_mw,
            "latency_ms": latency_ms, "reward": reward, "piv": piv_arr, "u_pir": u_pir_arr,
            "snr_db": snr_db, "arm": [arms[a].name for a in chosen_arm_idx],
        })
        per_window_logs[policy.name] = df_log

        for regime in np.unique(ds.regime):
            mask = ds.regime == regime
            all_records.append(compute_window_metrics(
                policy=policy.name, regime=regime,
                hr_true_bpm=ds.hr_true_bpm[mask], hr_delivered_bpm=hr_delivered[mask],
                decode_success=decode_success[mask], energy_j=energy_j[mask], power_mw=power_mw[mask],
                latency_ms=latency_ms[mask], reward=reward[mask], piv=piv_arr[mask], u_pir=u_pir_arr[mask],
                deadline_ms=config.reward_weights.latency_deadline_ms,
                power_cap_mw=config.reward_weights.power_cap_mw,
            ))
        log(f"    -> {policy.name}: overall PDR={np.mean(decode_success):.3f}, "
            f"HR-MAE={np.mean(np.abs(ds.hr_true_bpm - hr_delivered)):.3f} bpm, "
            f"mean reward={np.mean(reward):.4f}, final battery={result['final_battery_frac']:.3f}")

    df_regime = aggregate_metrics(all_records)
    df_overall = overall_metrics(df_regime)

    # --- Steady-state metrics (second half of the arrival-order session) ---
    # A bandit's *full-session* metrics unavoidably include its early,
    # largely-exploratory rounds, which structurally disadvantages it
    # relative to non-adaptive baselines that have no learning curve at
    # all. Standard bandit-evaluation practice separates cumulative
    # (whole-session, includes exploration cost) from steady-state
    # (post-convergence, "what does the learned policy actually do")
    # performance; we report both, but steady-state is the fairer number
    # for comparing *decision quality* once the bandit has converged.
    n_total_local = n_total
    arrival_position = np.empty(n_total_local, dtype=np.int64)
    arrival_position[order] = np.arange(n_total_local)
    steady_mask = arrival_position >= (n_total_local // 2)

    steady_records: list[EvaluationMetrics] = []
    for policy in policies:
        df_log = per_window_logs[policy.name]
        m = steady_mask
        steady_records.append(compute_window_metrics(
            policy=policy.name, regime="steady_state_last50pct",
            hr_true_bpm=ds.hr_true_bpm[m], hr_delivered_bpm=df_log["hr_delivered"].to_numpy()[m],
            decode_success=df_log["decode_success"].to_numpy()[m], energy_j=df_log["energy_j"].to_numpy()[m],
            power_mw=df_log["power_mw"].to_numpy()[m], latency_ms=df_log["latency_ms"].to_numpy()[m],
            reward=df_log["reward"].to_numpy()[m], piv=df_log["piv"].to_numpy()[m],
            u_pir=df_log["u_pir"].to_numpy()[m],
            deadline_ms=config.reward_weights.latency_deadline_ms,
            power_cap_mw=config.reward_weights.power_cap_mw,
        ))
    df_overall_steady = aggregate_metrics(steady_records)
    log("    Steady-state (last 50% of session) summary:")
    log(df_overall_steady[["policy", "hr_mae_bpm", "pdr", "mean_energy_j", "mean_reward"]].to_string(index=False))

    return {
        "seed": config.seed, "n_total": n_total, "arms": arms, "policies": policies,
        "bandit_label": bandit_label, "df_regime": df_regime, "df_overall": df_overall,
        "df_overall_steady": df_overall_steady,
        "per_window_logs": per_window_logs, "reward_traces_arrival_order": reward_traces_arrival_order,
        "arm_pull_counts": arm_pull_counts,
    }


def _export_single_seed_outputs(result: dict, config: SimulationConfig, outdir: Path) -> None:
    """Write the tables/figures for one seed's `run_single_seed` result.

    The headline tables/figures (overall_metrics.csv, the LaTeX table, and
    the bar/Pareto plots) use *steady-state* metrics (second half of the
    session, i.e. post-convergence decision quality); full-session
    (cumulative, including the bandit's exploration cost) metrics are also
    written separately as `full_session_metrics.csv` for transparency.
    """
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    df_regime = result["df_regime"]
    df_overall_full = result["df_overall"]
    df_overall = result["df_overall_steady"]
    per_window_logs = result["per_window_logs"]
    arms, bandit_label = result["arms"], result["bandit_label"]

    df_regime.to_csv(outdir / "tables" / "per_regime_metrics.csv", index=False)
    df_overall.to_csv(outdir / "tables" / "overall_metrics.csv", index=False)
    df_overall_full.to_csv(outdir / "tables" / "full_session_metrics.csv", index=False)

    latex = dataframe_to_latex_table(
        df_overall,
        caption=("PIR-Framework: steady-state (post-convergence, last 50\\% of session) "
                 "evaluation across five control policies."),
        label="tab:pir_overall",
        bold_best={"hr_mae_bpm": "min", "event_f1": "max", "pdr": "max",
                   "mean_energy_j": "min", "mean_latency_ms": "min",
                   "deadline_success_rate": "max", "mean_reward": "max"},
    )
    (outdir / "tables" / "overall_metrics_table.tex").write_text(latex)

    plot_policy_comparison_bars(df_overall, "hr_mae_bpm", outdir / "figures" / "fig_hr_mae",
                                 ylabel="Heart-rate MAE (bpm)", lower_is_better=True)
    plot_policy_comparison_bars(df_overall, "pdr", outdir / "figures" / "fig_pdr", ylabel="Packet delivery ratio")
    plot_policy_comparison_bars(df_overall, "event_f1", outdir / "figures" / "fig_event_f1",
                                 ylabel="Clinical event-detection F1")
    plot_policy_comparison_bars(df_overall, "mean_reward", outdir / "figures" / "fig_reward", ylabel="Mean reward")
    plot_energy_latency_tradeoff(df_overall, outdir / "figures" / "fig_energy_latency_tradeoff",
                                  deadline_ms=config.reward_weights.latency_deadline_ms)
    plot_pareto_frontier(df_overall, outdir / "figures" / "fig_pareto_frontier",
                          power_cap_mw=config.reward_weights.power_cap_mw)

    all_snr = per_window_logs[bandit_label]["snr_db"].to_numpy()
    all_success = per_window_logs[bandit_label]["decode_success"].to_numpy()
    plot_pdr_vs_snr(all_snr, all_success, outdir / "figures" / "fig_pdr_vs_snr")

    plot_regret_curve(result["reward_traces_arrival_order"], outdir / "figures" / "fig_cumulative_reward")

    if bandit_label in result["arm_pull_counts"]:
        plot_arm_allocation(
            [a.name for a in arms], result["arm_pull_counts"][bandit_label],
            outdir / "figures" / "fig_arm_allocation", title=f"{bandit_label}: learned arm allocation",
        )

    for name, df_log in per_window_logs.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        df_log.to_csv(outdir / "tables" / f"per_window_log_{safe_name}.csv", index=False)


def run_evaluation(config: SimulationConfig) -> dict:
    """Single-seed convenience entry point (used by the CLI's default mode):
    runs one seed and writes its tables/figures to `config.outdir`."""
    outdir = Path(config.outdir)
    result = run_single_seed(config, verbose=True)

    print("[5/5] Aggregating metrics, exporting tables and figures ...")
    _export_single_seed_outputs(result, config, outdir)

    summary = {
        "n_windows": result["n_total"], "n_arms": len(result["arms"]),
        "bandit_type": config.bandit_type, "seed": config.seed,
        "overall_metrics_path": str(outdir / "tables" / "overall_metrics.csv"),
        "per_regime_metrics_path": str(outdir / "tables" / "per_regime_metrics.csv"),
        "latex_table_path": str(outdir / "tables" / "overall_metrics_table.tex"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return {"df_regime": result["df_regime"], "df_overall": result["df_overall"],
            "per_window_logs": result["per_window_logs"]}


# --------------------------------------------------------------------------
# Multi-seed harness: statistical-significance sweep with 95% CIs
# --------------------------------------------------------------------------

_METRIC_COLS = [
    "hr_mae_bpm", "event_f1", "pdr", "mean_energy_j", "mean_power_mw",
    "mean_latency_ms", "deadline_success_rate", "power_cap_success_rate",
    "mean_reward", "mean_piv", "mean_u_pir",
]


def _mean_ci95(values: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of a 95% CI across independent seed replicates.
    Uses a t-critical value for the (typically small) seed count rather
    than the z=1.96 normal approximation, matching standard practice for
    n < ~30 replicates."""
    from scipy import stats as sstats

    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    t_crit = float(sstats.t.ppf(0.975, df=n - 1))
    return mean, t_crit * sem


def run_multi_seed_evaluation(
    base_config: SimulationConfig, seeds: list[int], outdir: str | None = None,
) -> dict:
    """
    Runs the full five-policy comparison independently for each seed in
    `seeds` (each seed draws its own synthetic dataset, its own channel
    noise realizations, and -- for the bandit -- its own exploration
    trajectory), then aggregates:

      * per-policy overall metrics -> mean and 95% CI across seeds
        (`overall_metrics_mean.csv`, `overall_metrics_ci95.csv`, and a
        LaTeX table rendering both together);
      * per-round reward traces -> a confidence-banded convergence plot
        (`fig_cumulative_reward_ci`);
      * pooled decode-success-vs-SNR points for the bandit policy, and
        summed arm-pull counts, for the diagnostic figures;
      * a Pareto-frontier and energy-latency tradeoff plot computed from
        the across-seed mean metrics.

    This is the entry point that should be used to make any claim of
    statistical significance or a reproducible Pareto-optimal frontier --
    a single seed is a single (possibly favorable or unfavorable) draw of
    both the synthetic physiological data and the channel noise process.
    """
    outdir = Path(outdir or base_config.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    print(f"=== Multi-seed evaluation: {len(seeds)} seeds {seeds}, "
          f"{base_config.n_per_regime} windows/regime x 6 regimes = "
          f"{base_config.n_per_regime * 6} windows/seed ===")

    per_seed_results = []
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i + 1}/{len(seeds)}) ---")
        cfg = replace(base_config, seed=seed, outdir=str(outdir / f"seed_{seed}"))
        result = run_single_seed(cfg, verbose=True)
        _export_single_seed_outputs(result, cfg, Path(cfg.outdir))
        per_seed_results.append(result)

    print("\n=== Aggregating across seeds ===")
    policy_names = [p.name for p in per_seed_results[0]["policies"]]
    bandit_label = per_seed_results[0]["bandit_label"]
    arms = per_seed_results[0]["arms"]

    # --- Overall metrics: mean + 95% CI across seeds, per policy ---
    # Aggregated from each seed's *steady-state* (post-convergence) metrics
    # -- see `run_single_seed` -- so the headline comparison reflects
    # converged decision quality rather than being dragged down by the
    # bandit's early-session exploration cost.
    mean_rows, ci_rows = [], []
    for policy in policy_names:
        mean_row = {"policy": policy}
        ci_row = {"policy": policy}
        for col in _METRIC_COLS:
            vals = np.array([
                r["df_overall_steady"].loc[r["df_overall_steady"]["policy"] == policy, col].iloc[0]
                for r in per_seed_results
            ])
            mean, ci = _mean_ci95(vals)
            mean_row[col] = mean
            ci_row[col] = ci
        mean_rows.append(mean_row)
        ci_rows.append(ci_row)
    df_overall_mean = pd.DataFrame(mean_rows)
    df_overall_ci = pd.DataFrame(ci_rows)

    df_overall_mean.to_csv(outdir / "tables" / "overall_metrics_mean.csv", index=False)
    df_overall_ci.to_csv(outdir / "tables" / "overall_metrics_ci95.csv", index=False)

    # Full-session (cumulative, includes bandit exploration cost) means too,
    # for transparency / regret-style reporting alongside the steady-state
    # headline table.
    full_mean_rows = []
    for policy in policy_names:
        row = {"policy": policy}
        for col in _METRIC_COLS:
            vals = np.array([
                r["df_overall"].loc[r["df_overall"]["policy"] == policy, col].iloc[0]
                for r in per_seed_results
            ])
            row[col], _ = _mean_ci95(vals)
        full_mean_rows.append(row)
    pd.DataFrame(full_mean_rows).to_csv(outdir / "tables" / "full_session_metrics_mean.csv", index=False)

    latex = dataframe_to_latex_table(
        df_overall_mean,
        caption=(f"PIR-Framework: steady-state (post-convergence) evaluation across five control "
                 f"policies, mean $\\pm$ 95\\% CI over {len(seeds)} independent seeds "
                 f"({base_config.n_per_regime * 6} windows/seed)."),
        label="tab:pir_overall_multiseed",
        bold_best={"hr_mae_bpm": "min", "event_f1": "max", "pdr": "max",
                   "mean_energy_j": "min", "mean_latency_ms": "min",
                   "deadline_success_rate": "max", "mean_reward": "max"},
        ci_df=df_overall_ci,
    )
    (outdir / "tables" / "overall_metrics_table_multiseed.tex").write_text(latex)

    # --- Per-regime metrics: mean across seeds (CI omitted for brevity) ---
    per_regime_all = pd.concat([r["df_regime"] for r in per_seed_results], ignore_index=True)
    df_regime_mean = per_regime_all.groupby(["policy", "regime"], as_index=False)[_METRIC_COLS].mean()
    df_regime_mean.to_csv(outdir / "tables" / "per_regime_metrics_mean.csv", index=False)

    # --- Convergence plot with shaded 95% CI across seeds ---
    n_rounds = per_seed_results[0]["n_total"]
    traces_by_policy: dict[str, np.ndarray] = {}
    for policy in policy_names:
        stacked = np.stack([r["reward_traces_arrival_order"][policy] for r in per_seed_results], axis=0)
        traces_by_policy[policy] = stacked  # (n_seeds, n_rounds)
    plot_regret_curve_with_ci(traces_by_policy, outdir / "figures" / "fig_cumulative_reward_ci")

    # --- Pooled decode-success vs. SNR for the bandit, across all seeds ---
    pooled_snr = np.concatenate([r["per_window_logs"][bandit_label]["snr_db"].to_numpy() for r in per_seed_results])
    pooled_success = np.concatenate(
        [r["per_window_logs"][bandit_label]["decode_success"].to_numpy() for r in per_seed_results],
    )
    plot_pdr_vs_snr(pooled_snr, pooled_success, outdir / "figures" / "fig_pdr_vs_snr_pooled",
                     label=f"{bandit_label}, pooled over {len(seeds)} seeds")

    # --- Summed arm-pull counts for the bandit across seeds ---
    if bandit_label in per_seed_results[0]["arm_pull_counts"]:
        summed_pulls = np.sum(
            np.stack([r["arm_pull_counts"][bandit_label] for r in per_seed_results], axis=0), axis=0,
        )
        plot_arm_allocation(
            [a.name for a in arms], summed_pulls, outdir / "figures" / "fig_arm_allocation_pooled",
            title=f"{bandit_label}: learned arm allocation (summed over {len(seeds)} seeds)",
        )

    # --- Policy-comparison bars with 95% CI error bars, Pareto plots on mean metrics ---
    df_for_bars = df_overall_mean.copy()
    for col in _METRIC_COLS:
        df_for_bars[f"{col}__ci"] = df_overall_ci[col].to_numpy()

    plot_policy_comparison_bars(df_for_bars, "hr_mae_bpm", outdir / "figures" / "fig_hr_mae_ci",
                                 ylabel="Heart-rate MAE (bpm)", lower_is_better=True, ci_col="hr_mae_bpm__ci")
    plot_policy_comparison_bars(df_for_bars, "pdr", outdir / "figures" / "fig_pdr_ci",
                                 ylabel="Packet delivery ratio", ci_col="pdr__ci")
    plot_policy_comparison_bars(df_for_bars, "mean_reward", outdir / "figures" / "fig_reward_ci",
                                 ylabel="Mean reward", ci_col="mean_reward__ci")
    plot_energy_latency_tradeoff(df_overall_mean, outdir / "figures" / "fig_energy_latency_tradeoff_mean",
                                  deadline_ms=base_config.reward_weights.latency_deadline_ms)
    plot_pareto_frontier(df_overall_mean, outdir / "figures" / "fig_pareto_frontier_mean",
                          power_cap_mw=base_config.reward_weights.power_cap_mw)

    print("\n=== Multi-seed summary (mean over seeds) ===")
    print(df_overall_mean[["policy", "hr_mae_bpm", "pdr", "mean_energy_j", "mean_power_mw",
                            "mean_latency_ms", "deadline_success_rate", "mean_reward"]]
          .to_string(index=False))

    summary = {
        "n_seeds": len(seeds), "seeds": seeds, "n_windows_per_seed": n_rounds,
        "n_arms": len(arms), "bandit_type": base_config.bandit_type,
        "overall_metrics_mean_path": str(outdir / "tables" / "overall_metrics_mean.csv"),
        "overall_metrics_ci95_path": str(outdir / "tables" / "overall_metrics_ci95.csv"),
        "latex_table_path": str(outdir / "tables" / "overall_metrics_table_multiseed.tex"),
    }
    (outdir / "summary_multiseed.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    return {
        "df_overall_mean": df_overall_mean, "df_overall_ci": df_overall_ci,
        "df_regime_mean": df_regime_mean, "per_seed_results": per_seed_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PIR-Framework comparative evaluation.")
    parser.add_argument("--n-per-regime", type=int, default=200,
                         help="Windows per degradation regime (x6 regimes = windows/seed).")
    parser.add_argument("--bandit", choices=["linucb", "thompson"], default="linucb")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--n-seeds", type=int, default=1,
                         help="If > 1, runs a multi-seed sweep (seed, seed+1, ..., seed+n_seeds-1) "
                              "and aggregates mean +/- 95%% CI across seeds.")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    config = SimulationConfig(
        n_per_regime=args.n_per_regime, bandit_type=args.bandit, seed=args.seed, outdir=args.outdir,
    )
    if args.n_seeds > 1:
        seeds = [args.seed + i for i in range(args.n_seeds)]
        run_multi_seed_evaluation(config, seeds, outdir=args.outdir)
    else:
        run_evaluation(config)


if __name__ == "__main__":
    main()
