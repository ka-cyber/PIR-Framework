"""Real fetal-ECG data path for the PIR framework.

This module adapts PIR to the PhysioNet Abdominal and Direct Fetal ECG
Database.  It deliberately does NOT pretend that this ECG-only dataset
contains PPG.  Instead it computes a multi-electrode ECG PIV variant from
four abdominal leads and evaluates it against the database's independent,
cardiologist-verified direct-fetal-ECG R-wave annotations.

PIV-ECG (multi-lead adaptation)
-------------------------------
For each abdominal lead, a periodicity SQI is computed as the peak normalized
autocorrelation over physiologically plausible fetal RR lags.  Per-lead FHR
is the lag corresponding to that peak.  The multi-lead quality term is the
presence-weighted arithmetic mean of per-lead SQIs.  A quality-weighted FHR
consensus is formed, and the disagreement penalty is the mean absolute
per-lead deviation from that consensus, normalized by ``hr_range_bpm``.

    PIV_ECG = clip(w_quality * mean_present(SQI_l)
                   - w_agreement * mean_present(abs(HR_l - HR_consensus))
                     / hr_range_bpm, 0, 1)

This is an explicitly labelled research adaptation, not the original ECG/PPG
PIV and not a clinically validated score.  Reference R-wave annotations are
used ONLY to calculate outcome metrics (reference FHR and estimation error);
they are never used as inputs to PIV or to the contextual-bandit controller.

The controller integration uses the four-lead PIV as the medical utility
input while two lead-pair summaries are placed in the framework's legacy
(two-modality) context slots.  These slots represent abdominal lead groups,
not ECG and PPG modalities.  Wireless losses and energy/latency remain
simulated because the supplied dataset contains no radio or device telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt


class MultiLeadPIVResult(NamedTuple):
    piv: np.ndarray
    sqi_combined: np.ndarray
    hr_consensus_bpm: np.ndarray
    disagreement_bpm: np.ndarray


def compute_multilead_piv(
    sqi: np.ndarray,
    hr_bpm: np.ndarray,
    present: np.ndarray | None = None,
    *,
    w_quality: float = 0.75,
    w_agreement: float = 0.25,
    hr_range_bpm: float = 140.0,
) -> MultiLeadPIVResult:
    """Compute the multi-electrode ECG adaptation of PIV.

    Inputs may be shape ``(n_leads,)`` for one window or ``(n_windows,
    n_leads)`` for a batch. ``present`` is a physical channel-availability
    mask; it is separate from a low/zero SQI.
    """
    sqi_arr = np.asarray(sqi, dtype=np.float64)
    hr_arr = np.asarray(hr_bpm, dtype=np.float64)
    if sqi_arr.ndim == 1:
        sqi_arr = sqi_arr[None, :]
        hr_arr = hr_arr[None, :]
        squeeze = True
    elif sqi_arr.ndim == 2:
        squeeze = False
    else:
        raise ValueError("sqi and hr_bpm must be one- or two-dimensional arrays")
    if hr_arr.shape != sqi_arr.shape:
        raise ValueError("sqi and hr_bpm must have identical shapes")
    if present is None:
        present_arr = np.ones_like(sqi_arr, dtype=bool)
    else:
        present_arr = np.asarray(present, dtype=bool)
        if present_arr.ndim == 1:
            present_arr = present_arr[None, :]
        if present_arr.shape != sqi_arr.shape:
            raise ValueError("present must match the shape of sqi and hr_bpm")
    if hr_range_bpm <= 0:
        raise ValueError("hr_range_bpm must be positive")
    if w_quality < 0 or w_agreement < 0:
        raise ValueError("PIV weights must be nonnegative")

    sqi_arr = np.clip(np.nan_to_num(sqi_arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    available_count = present_arr.sum(axis=1)
    sqi_sum = np.where(present_arr, sqi_arr, 0.0).sum(axis=1)
    sqi_combined = np.divide(
        sqi_sum, available_count, out=np.zeros_like(sqi_sum), where=available_count > 0,
    )

    # The consensus uses SQI as confidence weights; a physically present but
    # uninformative lead contributes zero weight to the HR estimate.
    valid_hr = present_arr & np.isfinite(hr_arr)
    hr_weights = np.where(valid_hr, sqi_arr, 0.0)
    weight_sum = hr_weights.sum(axis=1)
    safe_weights = np.where(weight_sum[:, None] > 1e-12, hr_weights, valid_hr.astype(float))
    safe_sum = safe_weights.sum(axis=1)
    hr_safe = np.where(valid_hr, hr_arr, 0.0)
    hr_consensus = np.divide(
        np.sum(safe_weights * hr_safe, axis=1), safe_sum,
        out=np.full(sqi_arr.shape[0], np.nan), where=safe_sum > 0,
    )

    abs_dev = np.where(valid_hr, np.abs(hr_arr - hr_consensus[:, None]), 0.0)
    n_valid_hr = valid_hr.sum(axis=1)
    disagreement = np.divide(
        abs_dev.sum(axis=1), n_valid_hr,
        out=np.zeros(sqi_arr.shape[0]), where=n_valid_hr > 0,
    )
    piv = np.clip(
        w_quality * sqi_combined - w_agreement * disagreement / hr_range_bpm,
        0.0, 1.0,
    )
    piv = np.where(available_count > 0, piv, 0.0)

    if squeeze:
        return MultiLeadPIVResult(
            piv=piv[0], sqi_combined=sqi_combined[0],
            hr_consensus_bpm=hr_consensus[0], disagreement_bpm=disagreement[0],
        )
    return MultiLeadPIVResult(piv, sqi_combined, hr_consensus, disagreement)


def estimate_fetal_rate_sqi(
    segment: np.ndarray,
    fs: float,
    *,
    band_hz: tuple[float, float] = (20.0, 100.0),
    fetal_hr_range_bpm: tuple[float, float] = (105.0, 180.0),
    signal_is_filtered: bool = False,
) -> tuple[float, float]:
    """Estimate per-lead fetal rate and a periodicity-based SQI for one window.

    The SQI is the maximum overlap-corrected normalized autocorrelation over
    the fetal RR-lag range. It is a transparent periodicity proxy, not the
    repository's Orphanidou-style template SQI and not a validated clinical
    signal-quality metric.
    """
    x = np.asarray(segment, dtype=np.float64)
    if x.ndim != 1 or len(x) < int(fs * 3):
        return float("nan"), 0.0
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(x) < 1e-12:
        return float("nan"), 0.0

    lo, hi = band_hz
    if not (0 < lo < hi < fs / 2):
        raise ValueError("band_hz must satisfy 0 < low < high < Nyquist")
    if signal_is_filtered:
        filtered = x.copy()
    else:
        sos = butter(3, [lo, hi], btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, x)
    filtered = filtered - np.mean(filtered)
    if np.std(filtered) < 1e-12:
        return float("nan"), 0.0

    ac = correlate(filtered, filtered, mode="full", method="fft")[len(filtered) - 1:]
    lag_min = max(1, int(np.floor(fs * 60.0 / fetal_hr_range_bpm[1])))
    lag_max = min(len(filtered) - 2, int(np.ceil(fs * 60.0 / fetal_hr_range_bpm[0])))
    if lag_max <= lag_min or ac[0] <= 0:
        return float("nan"), 0.0
    lags = np.arange(lag_min, lag_max + 1)
    # Correct for decreasing overlap at longer lags.
    ac_norm = ac[lags] / (ac[0] * (len(filtered) - lags) / len(filtered) + 1e-18)
    idx = int(np.argmax(ac_norm))
    peak = float(np.clip(ac_norm[idx], 0.0, 1.0))
    rate = float(60.0 * fs / lags[idx])
    return rate, peak


@dataclass
class FetalMultiLeadMedicalState:
    """Adapter for passing a multi-lead ECG PIV into the existing PIR engine.

    The first two context features are the mean SQI of abdomen lead pairs
    (1,2) and (3,4). The next two are physical availability flags. They occupy
    the shape of the framework's original ECG/PPG medical block but are NOT
    claimed to be ECG and PPG modalities.
    """
    group1_sqi: np.ndarray
    group2_sqi: np.ndarray
    piv: np.ndarray

    def as_array(self) -> np.ndarray:
        g1 = np.asarray(self.group1_sqi, dtype=np.float64)
        g2 = np.asarray(self.group2_sqi, dtype=np.float64)
        p = np.asarray(self.piv, dtype=np.float64)
        ones = np.ones_like(g1)
        return np.stack([g1, g2, ones, ones, p], axis=-1)


def read_edf_record(record_path: str | Path) -> tuple[np.ndarray, float, list[str], np.ndarray]:
    """Read one EDF record with MNE; return signals, fs, names, QRS times.

    The direct fetal ECG is retained as the reference channel. Its verified
    QRS annotations are used only for evaluation, never for PIV estimation.
    """
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError("Reading EDF records requires `pip install mne`.") from exc

    path = Path(record_path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    names = list(raw.ch_names)
    required = ["Direct_1", "Abdomen_1", "Abdomen_2", "Abdomen_3", "Abdomen_4"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path.name} is missing required channels: {missing}")
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    qrs_mask = np.char.upper(descriptions) == "QRS"
    qrs_times = np.asarray(raw.annotations.onset, dtype=np.float64)[qrs_mask]
    if len(qrs_times) < 5:
        raise ValueError(f"{path.name} does not contain usable QRS reference annotations")
    return raw.get_data(), float(raw.info["sfreq"]), names, qrs_times


def reference_hr_from_qrs(qrs_times_s: np.ndarray, start_s: float, end_s: float) -> float:
    """Median reference FHR from annotated RR intervals whose midpoint is in a window."""
    qrs = np.asarray(qrs_times_s, dtype=np.float64)
    if len(qrs) < 2:
        return float("nan")
    rr = np.diff(qrs)
    mid = 0.5 * (qrs[:-1] + qrs[1:])
    use = (mid >= start_s) & (mid < end_s) & (rr > 0)
    if use.sum() < 3:
        return float("nan")
    return float(60.0 / np.median(rr[use]))
