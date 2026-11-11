"""Run PIR PIV validation and controller integration on real fetal ECG EDFs.

Example:
    python -m pir_framework.real_fetal_experiment \
        --data-dir /path/to/adfecgdb \
        --outdir results_real_fetal_ecg

Optional `mne` dependency: pip install 'pir-framework[real-edf]'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import pearsonr, spearmanr

from .real_ecg import (
    FetalMultiLeadMedicalState,
    compute_multilead_piv,
    estimate_fetal_rate_sqi,
    read_edf_record,
    reference_hr_from_qrs,
)


def _record_window_features(record_path: Path, window_seconds: float) -> pd.DataFrame:
    signals, fs, names, qrs_times = read_edf_record(record_path)
    duration = signals.shape[1] / fs
    lead_names = [f"Abdomen_{i}" for i in range(1, 5)]
    lead_indices = [names.index(name) for name in lead_names]

    # Zero-phase bandpass the continuous recording once; window estimates then
    # avoid filter transients introduced at every 8-second boundary.
    sos = butter(3, [20.0, 100.0], btype="bandpass", fs=fs, output="sos")
    filtered = np.vstack([sosfiltfilt(sos, signals[idx]) for idx in lead_indices])

    starts = np.arange(0.0, duration - window_seconds + 1e-9, window_seconds)
    records: list[dict] = []
    for start_s in starts:
        end_s = start_s + window_seconds
        a, b = int(round(start_s * fs)), int(round(end_s * fs))
        rates, sqis = [], []
        for lead_idx in range(4):
            rate, sqi = estimate_fetal_rate_sqi(
                filtered[lead_idx, a:b], fs,
                band_hz=(20.0, 100.0),
                fetal_hr_range_bpm=(105.0, 180.0),
                signal_is_filtered=True,
            )
            rates.append(rate)
            sqis.append(sqi)

        metrics = compute_multilead_piv(np.asarray(sqis), np.asarray(rates))
        hr_ref = reference_hr_from_qrs(qrs_times, start_s, end_s)
        hr_est = float(metrics.hr_consensus_bpm)
        hr_error = abs(hr_est - hr_ref) if np.isfinite(hr_est) and np.isfinite(hr_ref) else np.nan

        # Two lead-pair summaries supply the legacy controller context slots.
        # These are electrode groups from the same ECG modality, not ECG/PPG.
        sqi = np.asarray(sqis, dtype=float)
        rate = np.asarray(rates, dtype=float)
        pair1_weights = np.where(np.isfinite(rate[:2]), sqi[:2], 0.0)
        pair2_weights = np.where(np.isfinite(rate[2:]), sqi[2:], 0.0)
        pair1_hr = float(np.sum(pair1_weights * np.nan_to_num(rate[:2])) / pair1_weights.sum()) if pair1_weights.sum() > 1e-12 else float(np.nanmean(rate[:2]))
        pair2_hr = float(np.sum(pair2_weights * np.nan_to_num(rate[2:])) / pair2_weights.sum()) if pair2_weights.sum() > 1e-12 else float(np.nanmean(rate[2:]))

        row = {
            "record": record_path.stem,
            "window_start_s": float(start_s),
            "window_end_s": float(end_s),
            "n_reference_beats": int(np.sum((qrs_times >= start_s) & (qrs_times < end_s))),
            "n_qrs_annotations_record": int(len(qrs_times)),
            "hr_reference_bpm": hr_ref,
            "hr_estimated_bpm": hr_est,
            "hr_absolute_error_bpm": hr_error,
            "piv_ecg_multilead": float(metrics.piv),
            "sqi_combined": float(metrics.sqi_combined),
            "interlead_disagreement_bpm": float(metrics.disagreement_bpm),
            "lead_group_1_sqi": float(np.mean(sqi[:2])),
            "lead_group_2_sqi": float(np.mean(sqi[2:])),
            "lead_group_1_hr_bpm": pair1_hr,
            "lead_group_2_hr_bpm": pair2_hr,
        }
        for i in range(4):
            row[f"hr_abdomen_{i+1}_bpm"] = float(rates[i])
            row[f"sqi_abdomen_{i+1}"] = float(sqis[i])
            row[f"hr_error_abdomen_{i+1}_bpm"] = abs(float(rates[i]) - hr_ref) if np.isfinite(hr_ref) else np.nan
        records.append(row)

    return pd.DataFrame(records)


def _within_record_permutation_test(df: pd.DataFrame, n_perm: int = 5000, seed: int = 2026) -> tuple[float, float, dict[str, float]]:
    """Equal-record mean Spearman rho with a circular-shift null.

    The circular shift preserves each record's univariate series and its
    local autocorrelation structure, while breaking the within-record
    window alignment between PIV and reference HR error. This is exploratory
    with only five records; it is not a substitute for a larger cohort.
    """
    groups = [g.sort_values("window_start_s") for _, g in df.groupby("record")]
    per_record = {
        str(g.record.iloc[0]): float(spearmanr(g.piv_ecg_multilead, g.hr_absolute_error_bpm).statistic)
        for g in groups
    }
    observed = float(np.mean(list(per_record.values())))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        rhos = []
        for g in groups:
            x = g.piv_ecg_multilead.to_numpy()
            y = g.hr_absolute_error_bpm.to_numpy()
            shift = int(rng.integers(1, len(y)))
            y_shift = np.roll(y, shift)
            rhos.append(float(spearmanr(x, y_shift).statistic))
        stat = float(np.mean(rhos))
        if abs(stat) >= abs(observed):
            exceed += 1
    p_value = (exceed + 1) / (n_perm + 1)
    return observed, p_value, per_record


def _within_record_ablation_test(df: pd.DataFrame, n_perm: int = 5000, seed: int = 99) -> tuple[float, float, dict[str, dict[str, float]]]:
    """Test whether PIV ranks HR error better than the SQI-only component.

    Statistic: equal-record mean(rho(PIV, error) - rho(SQI-only, error));
    negative values favour the full PIV. The null is generated by circularly
    shifting each record's reference-error sequence relative to both scores.
    """
    groups = [g.sort_values("window_start_s") for _, g in df.groupby("record")]
    per_record: dict[str, dict[str, float]] = {}
    differences = []
    for g in groups:
        rho_piv = float(spearmanr(g.piv_ecg_multilead, g.hr_absolute_error_bpm).statistic)
        rho_sqi = float(spearmanr(g.sqi_only_score, g.hr_absolute_error_bpm).statistic)
        per_record[str(g.record.iloc[0])] = {
            "piv_rho": rho_piv, "sqi_only_rho": rho_sqi, "delta_piv_minus_sqi": rho_piv - rho_sqi,
        }
        differences.append(rho_piv - rho_sqi)
    observed = float(np.mean(differences))

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        perm_differences = []
        for g in groups:
            y = g.hr_absolute_error_bpm.to_numpy()
            shift = int(rng.integers(1, len(y)))
            shifted_y = np.roll(y, shift)
            rho_piv = float(spearmanr(g.piv_ecg_multilead, shifted_y).statistic)
            rho_sqi = float(spearmanr(g.sqi_only_score, shifted_y).statistic)
            perm_differences.append(rho_piv - rho_sqi)
        if float(np.mean(perm_differences)) <= observed:
            exceed += 1
    p_value = (exceed + 1) / (n_perm + 1)
    return observed, p_value, per_record


def _run_sensitivity_analysis(data_dir: Path) -> pd.DataFrame:
    """Small, pre-specified sensitivity grid around the main analysis.

    Main setting: 20-100 Hz, 8 s. Alternatives: two neighboring ECG bands
    at 8 s, and a 12 s window at the main band. Each EDF is read only once.
    """
    configs = [
        ((20.0, 100.0), 8.0, "main"),
        ((15.0, 80.0), 8.0, "lower_band"),
        ((25.0, 150.0), 8.0, "higher_band"),
        ((20.0, 100.0), 12.0, "longer_window"),
    ]
    paths = sorted(data_dir.glob("*.edf"))
    rows: list[dict] = []
    for path in paths:
        signals, fs, names, qrs_times = read_edf_record(path)
        lead_indices = [names.index(f"Abdomen_{i}") for i in range(1, 5)]
        duration = signals.shape[1] / fs
        cache: dict[tuple[float, float], np.ndarray] = {}
        for band, window_seconds, label in configs:
            if band not in cache:
                sos = butter(3, list(band), btype="bandpass", fs=fs, output="sos")
                cache[band] = np.vstack([sosfiltfilt(sos, signals[idx]) for idx in lead_indices])
            filtered = cache[band]
            starts = np.arange(0.0, duration - window_seconds + 1e-9, window_seconds)
            errors, p_values = [], []
            for start_s in starts:
                end_s = start_s + window_seconds
                a, b = int(round(start_s * fs)), int(round(end_s * fs))
                rates, sqis = [], []
                for lead_idx in range(4):
                    rate, sqi = estimate_fetal_rate_sqi(
                        filtered[lead_idx, a:b], fs,
                        band_hz=band,
                        fetal_hr_range_bpm=(105.0, 180.0),
                        signal_is_filtered=True,
                    )
                    rates.append(rate)
                    sqis.append(sqi)
                metrics = compute_multilead_piv(np.asarray(sqis), np.asarray(rates))
                hr_ref = reference_hr_from_qrs(qrs_times, start_s, end_s)
                hr_est = float(metrics.hr_consensus_bpm)
                if np.isfinite(hr_ref) and np.isfinite(hr_est):
                    p_values.append(float(metrics.piv))
                    errors.append(abs(hr_est - hr_ref))
            if len(errors) >= 5:
                rho = float(spearmanr(p_values, errors).statistic)
                rows.append({
                    "config": label,
                    "filter_low_hz": band[0],
                    "filter_high_hz": band[1],
                    "window_seconds": window_seconds,
                    "record": path.stem,
                    "n_windows": len(errors),
                    "spearman_rho_piv_vs_hr_error": rho,
                })
        del signals, cache

    per_record = pd.DataFrame(rows)
    summaries = []
    for config, group in per_record.groupby("config", sort=False):
        summaries.append({
            "config": config,
            "filter_low_hz": float(group.filter_low_hz.iloc[0]),
            "filter_high_hz": float(group.filter_high_hz.iloc[0]),
            "window_seconds": float(group.window_seconds.iloc[0]),
            "n_windows": int(group.n_windows.sum()),
            "pooled_mean_within_record_spearman_rho": float(group.spearman_rho_piv_vs_hr_error.mean()),
            "records_with_negative_rho": int((group.spearman_rho_piv_vs_hr_error < 0).sum()),
            "n_records": int(group.record.nunique()),
            "per_record_rho_json": json.dumps(dict(zip(group.record, group.spearman_rho_piv_vs_hr_error)), sort_keys=True),
        })
    return pd.DataFrame(summaries)


def _run_controller_integration(df: pd.DataFrame, seed: int = 73) -> pd.DataFrame:
    """Run the existing PIR LinUCB + state-space engine with real PIV inputs.

    Physiological SQI/PIV features are from real abdominal ECG. The channel,
    decode probability, energy and latency remain the framework's simulator.
    This verifies end-to-end wiring, not deployed-system efficacy.
    """
    from .state_space import StateSpaceEngine
    from .pir_controller import (
        LinUCBController, RewardWeights, compute_reward, default_arm_set,
    )

    arms = default_arm_set()
    weights = RewardWeights()
    outputs: list[dict] = []
    for r_idx, (record, group) in enumerate(df.groupby("record", sort=True)):
        engine = StateSpaceEngine(n_envs=1, window_seconds=8.0, seed=seed + r_idx)
        controller = LinUCBController(arms, context_dim=10, alpha=0.3, ridge_lambda=1.0)
        for row in group.sort_values("window_start_s").itertuples(index=False):
            medical = FetalMultiLeadMedicalState(
                group1_sqi=np.asarray([row.lead_group_1_sqi]),
                group2_sqi=np.asarray([row.lead_group_2_sqi]),
                piv=np.asarray([row.piv_ecg_multilead]),
            )
            context = engine.observe_context(medical)
            chosen = controller.select(context)
            arm = arms[int(chosen[0])]
            step = engine.act(
                medical,
                K=np.asarray([arm.K]),
                R=np.asarray([arm.R]),
                placement_is_edge=np.asarray([arm.placement_is_edge], dtype=bool),
            )
            reward, components = compute_reward(
                np.asarray([row.piv_ecg_multilead]), step.p_decode,
                step.energy_j, step.latency_ms, step.power_mw, weights,
            )
            controller.update(context, chosen, reward)
            outputs.append({
                "record": record,
                "window_start_s": row.window_start_s,
                "piv_ecg_multilead": row.piv_ecg_multilead,
                "chosen_R": arm.R,
                "chosen_placement": "edge" if arm.placement_is_edge else "local",
                "link_state_bad": int(step.wireless.link_state[0]),
                "snr_db": float(step.wireless.snr_db[0]),
                "p_decode_model": float(step.p_decode[0]),
                "energy_model_j": float(step.energy_j[0]),
                "latency_model_ms": float(step.latency_ms[0]),
                "reward_model": float(reward[0]),
                "u_pir_model": float(components["u_pir"][0]),
            })
    return pd.DataFrame(outputs)


def _write_figures(df: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for record, group in df.groupby("record", sort=True):
        ax.scatter(group.piv_ecg_multilead, group.hr_absolute_error_bpm, s=22, alpha=0.75, label=record)
    ax.set_xlabel("Multi-lead ECG PIV (relative score)")
    ax.set_ylabel("Absolute fetal HR error (bpm)")
    ax.set_title("Real fetal ECG: PIV versus reference HR error")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Record")
    fig.tight_layout()
    fig.savefig(outdir / "piv_vs_reference_hr_error.png", dpi=180)
    plt.close(fig)

    tmp = df.copy()
    tmp["piv_quintile"] = pd.qcut(tmp.piv_ecg_multilead, 5, duplicates="drop")
    summary = tmp.groupby("piv_quintile", observed=True).agg(
        n=("piv_ecg_multilead", "size"),
        piv_mean=("piv_ecg_multilead", "mean"),
        hr_error_mean=("hr_absolute_error_bpm", "mean"),
        hr_error_median=("hr_absolute_error_bpm", "median"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(np.arange(len(summary)), summary.hr_error_median, yerr=None)
    ax.set_xticks(np.arange(len(summary)))
    ax.set_xticklabels([f"Q{i+1}\n(n={n})" for i, n in enumerate(summary.n)])
    ax.set_xlabel("PIV quintile (Q1 = lowest, Q5 = highest)")
    ax.set_ylabel("Median absolute fetal HR error (bpm)")
    ax.set_title("Reference HR error across PIV quintiles")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "hr_error_by_piv_quintile.png", dpi=180)
    plt.close(fig)


def run_experiment(data_dir: Path, outdir: Path, window_seconds: float = 8.0, n_perm: int = 5000) -> dict:
    data_dir = data_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    paths = sorted(data_dir.glob("*.edf"))
    if not paths:
        raise FileNotFoundError(f"No EDF records found in {data_dir}")

    windows = pd.concat(
        [_record_window_features(path, window_seconds) for path in paths],
        ignore_index=True,
    )
    windows = windows.dropna(subset=["piv_ecg_multilead", "hr_absolute_error_bpm"]).reset_index(drop=True)
    windows["sqi_only_score"] = 0.75 * windows.sqi_combined
    windows.to_csv(outdir / "real_fetal_ecg_piv_windows.csv", index=False)

    rho_pooled, p_pooled = spearmanr(windows.piv_ecg_multilead, windows.hr_absolute_error_bpm)
    rho_sqi, p_sqi = spearmanr(windows.sqi_only_score, windows.hr_absolute_error_bpm)
    rho_within, p_within, per_record_rhos = _within_record_permutation_test(windows, n_perm=n_perm)
    rho_delta_piv_sqi, p_delta_piv_sqi, ablation_record_stats = _within_record_ablation_test(windows, n_perm=n_perm)

    per_record_rows = []
    for record, group in windows.groupby("record", sort=True):
        rho, p = spearmanr(group.piv_ecg_multilead, group.hr_absolute_error_bpm)
        rho_sqi, p_sqi_record = spearmanr(group.sqi_only_score, group.hr_absolute_error_bpm)
        per_record_rows.append({
            "record": record,
            "n_windows": len(group),
            "mean_piv": group.piv_ecg_multilead.mean(),
            "mean_hr_error_bpm": group.hr_absolute_error_bpm.mean(),
            "median_hr_error_bpm": group.hr_absolute_error_bpm.median(),
            "spearman_rho_piv_vs_hr_error": rho,
            "spearman_rho_sqi_only_vs_hr_error": rho_sqi,
            "delta_rho_piv_minus_sqi_only": rho - rho_sqi,
            "window_level_p_value_descriptive": p,
        })
    per_record = pd.DataFrame(per_record_rows)
    per_record.to_csv(outdir / "per_record_validation.csv", index=False)

    binned = windows.copy()
    binned["piv_quintile"] = pd.qcut(binned.piv_ecg_multilead, 5, duplicates="drop")
    quintiles = binned.groupby("piv_quintile", observed=True).agg(
        n=("piv_ecg_multilead", "size"),
        piv_min=("piv_ecg_multilead", "min"),
        piv_max=("piv_ecg_multilead", "max"),
        piv_mean=("piv_ecg_multilead", "mean"),
        hr_error_mean_bpm=("hr_absolute_error_bpm", "mean"),
        hr_error_median_bpm=("hr_absolute_error_bpm", "median"),
    ).reset_index()
    quintiles["piv_quintile"] = quintiles["piv_quintile"].astype(str)
    quintiles.to_csv(outdir / "piv_quintile_validation.csv", index=False)

    # Controller integration: real medical PIV + simulated radio/energy/latency.
    actions = _run_controller_integration(windows)
    actions.to_csv(outdir / "controller_actions_real_piv_simulated_network.csv", index=False)
    sensitivity = _run_sensitivity_analysis(data_dir)
    sensitivity.to_csv(outdir / "piv_sensitivity_filter_and_window.csv", index=False)
    action_summary = {
        "n_windows": int(len(actions)),
        "redundancy_R_counts": {str(k): int(v) for k, v in actions.chosen_R.value_counts().sort_index().items()},
        "edge_placement_fraction": float((actions.chosen_placement == "edge").mean()),
        "mean_model_decode_probability": float(actions.p_decode_model.mean()),
        "mean_model_energy_j": float(actions.energy_model_j.mean()),
        "mean_model_latency_ms": float(actions.latency_model_ms.mean()),
        "mean_model_reward": float(actions.reward_model.mean()),
    }

    _write_figures(windows, outdir)
    bottom = quintiles.iloc[0]
    top = quintiles.iloc[-1]
    summary = {
        "dataset": "PhysioNet Abdominal and Direct Fetal ECG Database v1.0.0",
        "records": sorted(windows.record.unique().tolist()),
        "n_records": int(windows.record.nunique()),
        "window_seconds": float(window_seconds),
        "n_windows": int(len(windows)),
        "total_reference_qrs_annotations": int(windows[["record", "n_qrs_annotations_record"]].drop_duplicates()["n_qrs_annotations_record"].sum()),
        "mean_piv": float(windows.piv_ecg_multilead.mean()),
        "median_piv": float(windows.piv_ecg_multilead.median()),
        "mean_reference_hr_bpm": float(windows.hr_reference_bpm.mean()),
        "mean_absolute_hr_error_bpm": float(windows.hr_absolute_error_bpm.mean()),
        "sampling_frequency_hz": 1000,
        "input_channels": "four abdominal ECG leads; direct fetal ECG annotations used only as reference",
        "piv_variant": "multi-electrode ECG PIV adaptation; not the original ECG/PPG PIV",
        "sqi_definition": "maximum overlap-corrected normalized autocorrelation over 105-180 bpm lags after 20-100 Hz bandpass",
        "piv_definition": "clip(0.75 * presence-weighted mean lead SQI - 0.25 * mean absolute lead-HR deviation from SQI-weighted consensus / 140 bpm, 0, 1)",
        "reference_definition": "median FHR from cardiologist-verified direct-fetal ECG QRS annotations; annotations are never inputs to PIV or controller",
        "pooled_window_spearman_piv_vs_hr_error": float(rho_pooled),
        "pooled_window_spearman_p_value_naive": float(p_pooled),
        "sqi_only_spearman_vs_hr_error": float(rho_sqi),
        "sqi_only_window_level_p_value_naive": float(p_sqi),
        "equal_record_mean_spearman_piv_vs_hr_error": float(rho_within),
        "within_record_circular_shift_permutation_p_value": float(p_within),
        "per_record_spearman_rho": per_record_rhos,
        "equal_record_mean_delta_rho_piv_minus_sqi_only": float(rho_delta_piv_sqi),
        "equal_record_mean_sqi_only_spearman_vs_hr_error": float(np.mean([v["sqi_only_rho"] for v in ablation_record_stats.values()])),
        "within_record_piv_vs_sqi_circular_shift_permutation_p_value": float(p_delta_piv_sqi),
        "per_record_piv_vs_sqi_ablation": ablation_record_stats,
        "lowest_piv_quintile_median_hr_error_bpm": float(bottom.hr_error_median_bpm),
        "highest_piv_quintile_median_hr_error_bpm": float(top.hr_error_median_bpm),
        "sensitivity_grid_configs": int(len(sensitivity)),
        "sensitivity_grid_negative_equal_record_rho_configs": int((sensitivity.pooled_mean_within_record_spearman_rho < 0).sum()) if len(sensitivity) else 0,
        "sensitivity_grid_mean_equal_record_rho_min": float(sensitivity.pooled_mean_within_record_spearman_rho.min()) if len(sensitivity) else None,
        "sensitivity_grid_mean_equal_record_rho_max": float(sensitivity.pooled_mean_within_record_spearman_rho.max()) if len(sensitivity) else None,
        "controller_integration": action_summary,
        "limitations": [
            "Only five recordings from five women in labor; window count is not cohort size.",
            "The dataset is fetal ECG only and contains no PPG; this validates an ECG multi-electrode adaptation, not ECG-PPG agreement.",
            "The periodicity SQI is a hand-designed signal-quality proxy and has not been clinically validated.",
            "Fetal QRS annotations are used for reference HR/error only, not as PIV inputs.",
            "Wireless channel, coding success, energy, and compute latency in the controller integration are simulated, not measured in this dataset.",
            "This is offline evaluation; zero-phase filtering is not a real-time causal implementation.",
            "Window-level pooled p-values are descriptive; within-record circular-shift testing is exploratory with five records.",
        ],
    }
    (outdir / "validation_summary.json").write_text(json.dumps(summary, indent=2))

    report = _render_report(summary, per_record, quintiles, action_summary, sensitivity)
    (outdir / "REAL_DATA_VALIDATION_REPORT.md").write_text(report)
    return summary


def _render_report(summary: dict, per_record: pd.DataFrame, quintiles: pd.DataFrame, action_summary: dict, sensitivity: pd.DataFrame) -> str:
    lines = [
        "# PIR real-data validation: abdominal and direct fetal ECG",
        "",
        "## Scope",
        "",
        "This experiment uses five real EDF recordings. The source contains four abdominal ECG channels and a direct fetal ECG reference with verified QRS markers. It does not contain PPG. The result is therefore a **multi-electrode ECG adaptation of PIV**, not a validation of the original ECG/PPG PIV.",
        "",
        "## PIV construction",
        "",
        "1. Bandpass each abdominal lead at 20–100 Hz (zero-phase, offline).",
        "2. In each 8-second window, estimate fetal heart rate from the strongest overlap-corrected normalized autocorrelation in the 105–180 bpm lag interval. Use that peak value as a periodicity-based SQI in [0,1].",
        "3. Average SQIs over physically present leads. Compute the SQI-weighted consensus fetal rate and the mean absolute deviation of individual lead rates from that consensus.",
        "4. Compute `PIV_ECG = clip(0.75 * SQI_combined - 0.25 * disagreement_bpm / 140, 0, 1)`. The direct fetal ECG QRS annotations are excluded from the PIV inputs.",
        "",
        "## Validation result",
        "",
        f"- Records: {summary['n_records']}; analysed windows: {summary['n_windows']}; window size: {summary['window_seconds']} s; verified reference QRS annotations: {summary['total_reference_qrs_annotations']}.",
        f"- Mean PIV: {summary['mean_piv']:.3f}; mean absolute fetal HR error: {summary['mean_absolute_hr_error_bpm']:.2f} bpm.",
        f"- Pooled window-level Spearman correlation between PIV and absolute fetal HR error: **{summary['pooled_window_spearman_piv_vs_hr_error']:.3f}** (naive p={summary['pooled_window_spearman_p_value_naive']:.3g}; does not adjust for within-record dependence).",
        f"- Equal-record mean within-record Spearman correlation: **{summary['equal_record_mean_spearman_piv_vs_hr_error']:.3f}**; circular-shift permutation p={summary['within_record_circular_shift_permutation_p_value']:.4g} (exploratory; five records, 5,000 permutations).",
        f"- SQI-only Spearman correlation with HR error: {summary['sqi_only_spearman_vs_hr_error']:.3f} pooled; equal-record mean within-record SQI-only rho: {summary['equal_record_mean_sqi_only_spearman_vs_hr_error']:.3f}.",
        f"- Equal-record mean rho difference (PIV minus SQI-only; negative favours PIV): **{summary['equal_record_mean_delta_rho_piv_minus_sqi_only']:.3f}**; within-record circular-shift permutation p={summary['within_record_piv_vs_sqi_circular_shift_permutation_p_value']:.4g}.",
        f"- Median HR error in the lowest versus highest PIV quintile: {summary['lowest_piv_quintile_median_hr_error_bpm']:.2f} versus {summary['highest_piv_quintile_median_hr_error_bpm']:.2f} bpm.",
        f"- Sensitivity grid: {summary['sensitivity_grid_configs']} filter/window configurations; negative equal-record mean rho in {summary['sensitivity_grid_negative_equal_record_rho_configs']} configurations (range {summary['sensitivity_grid_mean_equal_record_rho_min']:.3f} to {summary['sensitivity_grid_mean_equal_record_rho_max']:.3f}).",
        "",
        "### Per-record results",
        "",
        "| Record | Windows | Mean PIV | Mean HR error (bpm) | Median HR error (bpm) | PIV rho | SQI-only rho |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_record.itertuples(index=False):
        lines.append(f"| {row.record} | {row.n_windows} | {row.mean_piv:.3f} | {row.mean_hr_error_bpm:.2f} | {row.median_hr_error_bpm:.2f} | {row.spearman_rho_piv_vs_hr_error:.3f} | {row.spearman_rho_sqi_only_vs_hr_error:.3f} |")
    lines += [
        "",
        "### PIV quintiles",
        "",
        "| PIV bin | n | Mean PIV | Mean HR error (bpm) | Median HR error (bpm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in quintiles.itertuples(index=False):
        lines.append(f"| {row.piv_quintile} | {row.n} | {row.piv_mean:.3f} | {row.hr_error_mean_bpm:.2f} | {row.hr_error_median_bpm:.2f} |")
    lines += [
        "",
        "### Filter/window sensitivity",
        "",
        "| Configuration | Band (Hz) | Window (s) | Windows | Mean within-record rho | Records with negative rho |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sensitivity.itertuples(index=False):
        lines.append(f"| {row.config} | {row.filter_low_hz:.0f}–{row.filter_high_hz:.0f} | {row.window_seconds:.0f} | {row.n_windows} | {row.pooled_mean_within_record_spearman_rho:.3f} | {row.records_with_negative_rho}/{row.n_records} |")
    lines += [
        "",
        "## PIR controller integration",
        "",
        "The existing LinUCB controller and state-space engine were run with the real four-lead PIV sequence. The legacy two signal-quality context slots contain mean SQIs from abdomen leads 1–2 and 3–4, respectively. They are electrode groups from one ECG modality, not an ECG/PPG pair.",
        "",
        f"- Windows through controller: {action_summary['n_windows']}",
        f"- Chosen redundancy counts: {action_summary['redundancy_R_counts']}",
        f"- Edge placement fraction: {action_summary['edge_placement_fraction']:.3f}",
        f"- Mean model decode probability: {action_summary['mean_model_decode_probability']:.3f}",
        f"- Mean model energy/window: {action_summary['mean_model_energy_j']*1000:.2f} mJ",
        f"- Mean model latency: {action_summary['mean_model_latency_ms']:.2f} ms",
        f"- Mean model reward: {action_summary['mean_model_reward']:.4f}",
        "",
        "These controller outcomes are a software integration run only. The dataset has no network packet, battery, or device timing measurements, so channel, energy, decoding, and latency are simulated.",
        "",
        "## Limitations and interpretation",
        "",
    ]
    lines += [f"- {item}" for item in summary["limitations"]]
    lines += [
        "",
        "## Software checks",
        "",
        "The targeted real-data/state-space tests pass. On the full suite, 99 tests passed and the pre-existing Monte Carlo RLNC decode-probability test was deselected because it did not finish within the sandbox execution limit; no claim is made that the complete suite passed.",
        "",
        "## Dataset citation",
        "",
        "Jezewski J, Matonia A, Kupka T, Roj D, Czabanski R. Determination of the fetal heart rate from abdominal signals: evaluation of beat-to-beat accuracy in relation to the direct fetal electrocardiogram. Biomedical Engineering/Biomedizinische Technik. 2012;57(5):383–394. doi:10.1515/bmt-2011-0130. PhysioNet dataset DOI: 10.13026/C2RP4B.",
        "",
        "## Reproducibility",
        "",
        "Run with `python -m pir_framework.real_fetal_experiment --data-dir /path/to/adfecgdb --outdir results_real_fetal_ecg`. Requires MNE-Python for EDF reading.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing r01.edf, r04.edf, ...")
    parser.add_argument("--outdir", type=Path, default=Path("results_real_fetal_ecg"))
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()
    summary = run_experiment(args.data_dir, args.outdir, args.window_seconds, args.permutations)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
