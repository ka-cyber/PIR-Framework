"""
Data pipeline: PhysioNet-format-compatible loading interface + a
physiologically-grounded synthetic ECG/PPG generator with a controllable
baseline-drift / motion-artifact / muscle-artifact injector, plus a
lightweight Orphanidou-style Signal Quality Index (SQI) estimator.

Real-data path
--------------
`PhysioNetRecordLoader` wraps the `wfdb` package (the standard PhysioNet
WFDB reader used by MIMIC-III waveform and PTB-XL) when it is installed and
local record files are available. It is written against the documented
`wfdb.rdrecord` API and is exercised by the test suite via a synthetic
in-memory WFDB-shaped record, so the interface is verified even in
environments (such as this one) with no network access to actually
download PhysioNet data.

Synthetic path (default; used by `evaluate.py`)
------------------------------------------------
No public dataset provides ECG and PPG with *independently controllable,
graded* degradation severity and known ground truth (the same limitation
documented by CardioFusion-AI, Section II-C). `SyntheticECGPPGGenerator`
therefore follows the identical controlled-degradation protocol: a
McSharry-style dynamical ECG model reduced to a sum-of-Gaussians
approximation (used here in place of a NeuroKit2 dependency, which is not
required for the framework to run), a matched PPG model driven by the same
heart rate, and five severity levels (clean/mild/moderate/severe/missing)
per modality, fully vectorized over a batch of windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


# --------------------------------------------------------------------------
# Real-data loading (PhysioNet / MIMIC-III / PTB-XL via WFDB)
# --------------------------------------------------------------------------


@dataclass
class PhysioNetRecordLoader:
    """
    Thin, dependency-guarded wrapper around `wfdb.rdrecord` for reading
    PhysioNet-format ECG/PPG records (the format used by MIMIC-III
    waveform matched-subset and PTB-XL exports).

    Usage
    -----
        loader = PhysioNetRecordLoader(record_dir="/path/to/records")
        sig, fs, channel_names = loader.load("p000001")

    If the `wfdb` package is not installed, `load` raises a clear
    `ImportError` rather than failing on an obscure attribute error, so
    calling code can fall back to `SyntheticECGPPGGenerator`.
    """

    record_dir: str

    def load(self, record_name: str) -> tuple[np.ndarray, float, list[str]]:
        try:
            import wfdb
        except ImportError as exc:  # pragma: no cover - exercised only without wfdb installed
            raise ImportError(
                "PhysioNetRecordLoader requires the 'wfdb' package "
                "(`pip install wfdb`) to read real MIMIC-III/PTB-XL records. "
                "Falling back to SyntheticECGPPGGenerator is recommended if "
                "wfdb is unavailable or the dataset has not been downloaded."
            ) from exc

        record = wfdb.rdrecord(str(Path(self.record_dir) / record_name))
        signal = np.asarray(record.p_signal, dtype=np.float64).T  # (channels, samples)
        fs = float(record.fs)
        channel_names = list(record.sig_name)
        return signal, fs, channel_names

    @staticmethod
    def select_channels(
        signal: np.ndarray, channel_names: list[str], ecg_keys=("ECG", "II", "MLII"), ppg_keys=("PLETH", "PPG"),
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Best-effort channel selection by common PhysioNet channel-name
        conventions (MIMIC-III uses 'II'/'MLII' for ECG and 'PLETH' for
        PPG; PTB-XL uses 12-lead ECG names)."""
        ecg = ppg = None
        for i, name in enumerate(channel_names):
            upper = name.upper()
            if ecg is None and any(k in upper for k in ecg_keys):
                ecg = signal[i]
            if ppg is None and any(k in upper for k in ppg_keys):
                ppg = signal[i]
        return ecg, ppg


# --------------------------------------------------------------------------
# Synthetic, physiologically-grounded ECG/PPG generation
# --------------------------------------------------------------------------


def _synthetic_ecg_beat(t: np.ndarray, hr_bpm: np.ndarray) -> np.ndarray:
    """Vectorized sum-of-asymmetric-Gaussians ECG approximation (P-QRS-T
    complex), driven by heart rate. `t` has shape (n_windows, n_samples) of
    time-since-last-beat-phase in [0, 1) (i.e. already phase-wrapped by the
    caller); `hr_bpm` has shape (n_windows, 1) for QRS-width scaling."""
    # (center_phase, amplitude, width) for P, Q, R, S, T waves, as fractions
    # of one cardiac cycle -- standard illustrative ECG-complex geometry.
    components = [
        (0.16, 0.15, 0.025),   # P wave
        (0.28, -0.10, 0.008),  # Q dip
        (0.30, 1.00, 0.010),   # R spike
        (0.32, -0.20, 0.008),  # S dip
        (0.55, 0.30, 0.040),   # T wave
    ]
    out = np.zeros_like(t)
    for center, amp, width in components:
        out += amp * np.exp(-0.5 * ((t - center) / width) ** 2)
    return out


def _synthetic_ppg_beat(t: np.ndarray) -> np.ndarray:
    """Vectorized PPG pulse approximation: a fast systolic upstroke Gaussian
    plus a slower, delayed diastolic (dicrotic) Gaussian, matching the
    qualitative morphology used throughout the PPG signal-processing
    literature (and CardioFusion-AI's systolic-peak-detection front end)."""
    systolic = np.exp(-0.5 * ((t - 0.18) / 0.06) ** 2)
    dicrotic = 0.35 * np.exp(-0.5 * ((t - 0.50) / 0.10) ** 2)
    return systolic + dicrotic


@dataclass
class DegradationConfig:
    """Severity-scaled degradation parameters, one field per artifact type,
    following the CardioFusion-AI Section II-C degradation protocol
    (clean=0, mild=1, moderate=2, severe=3, missing=4)."""

    gaussian_noise_std: tuple[float, ...] = (0.0, 0.03, 0.07, 0.15)
    baseline_wander_amp: tuple[float, ...] = (0.0, 0.05, 0.12, 0.25)
    baseline_wander_hz: float = 0.3
    muscle_artifact_amp: tuple[float, ...] = (0.0, 0.04, 0.10, 0.22)
    muscle_artifact_band_hz: tuple[float, float] = (20.0, 45.0)
    motion_burst_prob: tuple[float, ...] = (0.0, 0.05, 0.15, 0.30)
    motion_burst_amp: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0)


@dataclass
class SyntheticWindow:
    ecg: np.ndarray            # (N, n_samples)
    ppg: np.ndarray            # (N, n_samples)
    hr_true_bpm: np.ndarray    # (N,)
    ecg_level: np.ndarray      # (N,) int in {0,1,2,3,4}
    ppg_level: np.ndarray      # (N,) int in {0,1,2,3,4}
    fs: float


class SyntheticECGPPGGenerator:
    """
    Physiologically-grounded synthetic ECG/PPG generator with independently
    controllable, graded per-modality degradation, following the same
    protocol used to validate CardioFusion-AI's fusion architectures
    (clean/mild/moderate/severe/missing x2 modalities).

    Not a claim of clinical realism at the waveform-morphology level (a true
    McSharry dynamical-model or NeuroKit2 simulator would be more faithful);
    it is a controlled stand-in that reproduces the *structure* the rest of
    the framework needs (a genuine ground-truth heart rate, quality that
    degrades monotonically with severity, and complete-dropout conditions)
    without requiring network access to fetch PhysioNet data or an optional
    NeuroKit2 dependency.
    """

    def __init__(
        self, fs: float = 125.0, window_seconds: float = 8.0,
        degradation: DegradationConfig | None = None, seed: int | None = None,
    ) -> None:
        self.fs = fs
        self.window_seconds = window_seconds
        self.n_samples = int(round(fs * window_seconds))
        self.degradation = degradation or DegradationConfig()
        self.rng = np.random.default_rng(seed)

    def _cardiac_phase(self, hr_bpm: np.ndarray, phase0: np.ndarray) -> np.ndarray:
        """Phase-in-cycle in [0, 1) for every (window, sample), given a
        per-window heart rate and initial phase offset."""
        t = np.arange(self.n_samples) / self.fs  # (n_samples,)
        beat_freq_hz = hr_bpm / 60.0  # (N,)
        phase = (phase0[:, None] + beat_freq_hz[:, None] * t[None, :]) % 1.0
        return phase

    def generate_clean(self, n_windows: int, hr_range_bpm: tuple[float, float] = (50.0, 110.0)):
        hr_true = self.rng.uniform(*hr_range_bpm, size=n_windows)
        phase0_ecg = self.rng.uniform(0, 1, size=n_windows)
        phase0_ppg = phase0_ecg.copy()  # PPG pulse follows the same cardiac cycle as ECG

        phase_ecg = self._cardiac_phase(hr_true, phase0_ecg)
        phase_ppg = self._cardiac_phase(hr_true, phase0_ppg)

        ecg = _synthetic_ecg_beat(phase_ecg, hr_true[:, None])
        ppg = _synthetic_ppg_beat(phase_ppg)
        return ecg, ppg, hr_true

    def apply_degradation(self, signal: np.ndarray, level: np.ndarray, kind: str) -> np.ndarray:
        """Apply severity-scaled Gaussian noise + baseline wander + a
        high-frequency 'muscle artifact' band + randomly-timed motion
        bursts. `level` is an integer array in {0,1,2,3} (4=missing is
        handled separately by the caller, since it replaces the signal
        entirely with a structureless noise floor)."""
        cfg = self.degradation
        n_windows, n_samples = signal.shape
        out = signal.copy()
        t = np.arange(n_samples) / self.fs

        noise_std = np.array(cfg.gaussian_noise_std)[level]
        out += self.rng.normal(0.0, 1.0, size=out.shape) * noise_std[:, None]

        wander_amp = np.array(cfg.baseline_wander_amp)[level]
        wander_phase = self.rng.uniform(0, 2 * np.pi, size=n_windows)
        wander = np.sin(2 * np.pi * cfg.baseline_wander_hz * t[None, :] + wander_phase[:, None])
        out += wander_amp[:, None] * wander

        muscle_amp = np.array(cfg.muscle_artifact_amp)[level]
        band_low, band_high = cfg.muscle_artifact_band_hz
        muscle_noise = self.rng.normal(0.0, 1.0, size=out.shape)
        muscle_noise = _bandpass_fft(muscle_noise, self.fs, band_low, band_high)
        out += muscle_amp[:, None] * muscle_noise

        burst_prob = np.array(cfg.motion_burst_prob)[level]
        burst_amp = np.array(cfg.motion_burst_amp)[level]
        burst_mask = self.rng.random(n_windows) < burst_prob
        if np.any(burst_mask):
            n_bursts = int(np.sum(burst_mask))
            burst_center = self.rng.uniform(0.1, 0.9, size=n_bursts) * n_samples
            burst_width = self.rng.uniform(0.02, 0.08, size=n_bursts) * n_samples
            idx = np.arange(n_samples)[None, :]
            envelope = np.exp(-0.5 * ((idx - burst_center[:, None]) / burst_width[:, None]) ** 2)
            burst_signal = envelope * self.rng.normal(0.0, 1.0, size=envelope.shape)
            out[burst_mask] += burst_amp[burst_mask][:, None] * burst_signal
        return out

    def generate(
        self, n_windows: int, ecg_level: np.ndarray, ppg_level: np.ndarray,
        hr_range_bpm: tuple[float, float] = (50.0, 110.0),
    ) -> SyntheticWindow:
        """Generate `n_windows` windows with per-window degradation levels
        (0-3 graded, 4 = complete missing/dropout) supplied by the caller."""
        ecg_clean, ppg_clean, hr_true = self.generate_clean(n_windows, hr_range_bpm)

        ecg_graded_mask = ecg_level < 4
        ppg_graded_mask = ppg_level < 4
        ecg_out = ecg_clean.copy()
        ppg_out = ppg_clean.copy()

        if np.any(ecg_graded_mask):
            ecg_out[ecg_graded_mask] = self.apply_degradation(
                ecg_clean[ecg_graded_mask], ecg_level[ecg_graded_mask], kind="ecg",
            )
        if np.any(ppg_graded_mask):
            ppg_out[ppg_graded_mask] = self.apply_degradation(
                ppg_clean[ppg_graded_mask], ppg_level[ppg_graded_mask], kind="ppg",
            )

        missing_ecg = ecg_level == 4
        missing_ppg = ppg_level == 4
        if np.any(missing_ecg):
            ecg_out[missing_ecg] = self.rng.normal(0.0, 0.5, size=(int(np.sum(missing_ecg)), self.n_samples))
        if np.any(missing_ppg):
            ppg_out[missing_ppg] = self.rng.normal(0.0, 0.5, size=(int(np.sum(missing_ppg)), self.n_samples))

        return SyntheticWindow(
            ecg=ecg_out, ppg=ppg_out, hr_true_bpm=hr_true,
            ecg_level=ecg_level, ppg_level=ppg_level, fs=self.fs,
        )


def _bandpass_fft(signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Simple, fully-vectorized FFT band-pass filter (zeroes frequency bins
    outside [low_hz, high_hz]) used to synthesize a band-limited
    high-frequency 'muscle artifact' noise process."""
    n_samples = signal.shape[-1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum = np.fft.rfft(signal, axis=-1)
    spectrum *= mask[None, :]
    filtered = np.fft.irfft(spectrum, n=n_samples, axis=-1)
    # Re-normalize so the artifact's injected amplitude is comparable across
    # different (low_hz, high_hz) bands regardless of the resulting
    # narrower/wider effective bandwidth.
    scale = np.std(signal, axis=-1, keepdims=True) / (np.std(filtered, axis=-1, keepdims=True) + 1e-9)
    return filtered * scale


# --------------------------------------------------------------------------
# Lightweight Orphanidou-style Signal Quality Index
# --------------------------------------------------------------------------


@dataclass
class SQIResult:
    sqi: np.ndarray               # (N,) in [0, 1]; 0 if feasibility check fails
    hr_est_bpm: np.ndarray        # (N,) estimated heart rate from detected peaks
    passes_feasibility: np.ndarray  # (N,) bool


def estimate_sqi_and_hr(
    signal: np.ndarray, fs: float, hr_range_bpm: tuple[float, float] = (35.0, 200.0),
    max_pp_ratio: float = 3.0, max_plausible_bpm_for_peak_spacing: float = 130.0,
) -> SQIResult:
    """
    Two-stage signal-quality estimator, following Orphanidou et al. (2015)
    as implemented in CardioFusion-AI's `preprocessing/signal_quality.py`:

    Stage 1 (feasibility): peak detection + physiologically-plausible
        heart-rate range + bounded beat-to-beat interval ratio.
    Stage 2 (template correlation): build an adaptive per-window beat
        template by averaging fixed-length windows centered on every
        detected peak, then score quality as the mean Pearson correlation
        between the template and each individual beat window.

    Runs per-row (window) in a Python loop -- peak detection itself is
    inherently variable-length per window and not meaningfully
    vectorizable across windows with differing numbers of detected peaks --
    but every per-window computation is itself vectorized (NumPy), and the
    loop is over windows (typically tens to low thousands), not samples.
    """
    n_windows, n_samples = signal.shape
    sqi = np.zeros(n_windows)
    hr_est = np.full(n_windows, np.nan)
    passes = np.zeros(n_windows, dtype=bool)

    min_distance = max(1, int(fs * 60.0 / max_plausible_bpm_for_peak_spacing))
    for i in range(n_windows):
        x = signal[i]
        std = np.std(x)
        if std < 1e-8:
            continue
        # `prominence` (rather than raw `height`) rejects secondary
        # waveform features close to a larger peak -- e.g. the PPG
        # dicrotic notch -- that a pure height threshold can otherwise
        # double-count as a full beat, systematically doubling the
        # estimated heart rate.
        peaks, _ = find_peaks(x, distance=min_distance, prominence=0.35 * std)
        if len(peaks) < 3:
            continue

        pp_intervals_sec = np.diff(peaks) / fs
        if np.any(pp_intervals_sec <= 0):
            continue
        hr_bpm = 60.0 / np.mean(pp_intervals_sec)
        pp_ratio = float(np.max(pp_intervals_sec) / np.min(pp_intervals_sec))
        feasible = (hr_range_bpm[0] <= hr_bpm <= hr_range_bpm[1]) and (pp_ratio < max_pp_ratio)
        if not feasible:
            # No physiologically plausible estimate this window; leave
            # hr_est[i] as NaN so downstream code falls back to a prior
            # (e.g. the last known-good estimate) rather than propagating
            # a numerically-finite but physiologically meaningless number
            # derived from spurious noise peaks.
            continue

        half_w = int(round(np.median(np.diff(peaks)) / 2))
        half_w = max(half_w, 3)
        valid_peaks = peaks[(peaks - half_w >= 0) & (peaks + half_w < n_samples)]
        if len(valid_peaks) < 3:
            hr_est[i] = hr_bpm
            passes[i] = feasible
            continue
        beats = np.stack([x[p - half_w: p + half_w] for p in valid_peaks])
        template = beats.mean(axis=0)
        beats_c = beats - beats.mean(axis=1, keepdims=True)
        template_c = template - template.mean()
        num = beats_c @ template_c
        denom = np.linalg.norm(beats_c, axis=1) * (np.linalg.norm(template_c) + 1e-9) + 1e-9
        corrs = num / denom
        sqi[i] = float(np.clip(np.mean(corrs), 0.0, 1.0))
        hr_est[i] = hr_bpm
        passes[i] = feasible

    return SQIResult(sqi=sqi, hr_est_bpm=hr_est, passes_feasibility=passes)


# --------------------------------------------------------------------------
# End-to-end window builder used by evaluate.py
# --------------------------------------------------------------------------


REGIME_GRID = {
    "both_clean": (0, 0),
    "ecg_degraded": None,   # ECG in {1,2,3}, PPG = 0
    "ppg_degraded": None,   # ECG = 0, PPG in {1,2,3}
    "both_degraded": None,  # ECG in {1,2,3}, PPG in {1,2,3}
    "ecg_missing": (4, 0),
    "ppg_missing": (0, 4),
}


@dataclass
class PhysiologicalDataset:
    ecg: np.ndarray
    ppg: np.ndarray
    hr_true_bpm: np.ndarray
    ecg_level: np.ndarray
    ppg_level: np.ndarray
    regime: np.ndarray          # (N,) string labels
    sqi_ecg: np.ndarray
    sqi_ppg: np.ndarray
    hr_est_ecg: np.ndarray
    hr_est_ppg: np.ndarray
    present_ecg: np.ndarray
    present_ppg: np.ndarray
    fs: float


def build_dataset(
    n_per_regime: int, generator: SyntheticECGPPGGenerator | None = None, seed: int | None = None,
) -> PhysiologicalDataset:
    """Builds a balanced dataset spanning all six CardioFusion-AI-style
    degradation regimes (both_clean, ecg_degraded, ppg_degraded,
    both_degraded, ecg_missing, ppg_missing), computes SQI/heart-rate
    estimates per modality, and returns everything the downstream
    state-space / controller / evaluation code needs."""
    rng = np.random.default_rng(seed)
    gen = generator or SyntheticECGPPGGenerator(seed=seed)

    ecg_levels, ppg_levels, regimes = [], [], []

    def add(n, ecg_choices, ppg_choices, label):
        ecg_levels.append(rng.choice(ecg_choices, size=n))
        ppg_levels.append(rng.choice(ppg_choices, size=n))
        regimes.extend([label] * n)

    add(n_per_regime, [0], [0], "both_clean")
    add(n_per_regime, [1, 2, 3], [0], "ecg_degraded")
    add(n_per_regime, [0], [1, 2, 3], "ppg_degraded")
    add(n_per_regime, [1, 2, 3], [1, 2, 3], "both_degraded")
    add(n_per_regime, [4], [0], "ecg_missing")
    add(n_per_regime, [0], [4], "ppg_missing")

    ecg_level = np.concatenate(ecg_levels)
    ppg_level = np.concatenate(ppg_levels)
    regime = np.array(regimes)

    n_total = len(regime)
    perm = rng.permutation(n_total)
    ecg_level, ppg_level, regime = ecg_level[perm], ppg_level[perm], regime[perm]

    window = gen.generate(n_total, ecg_level, ppg_level)

    sqi_ecg_res = estimate_sqi_and_hr(window.ecg, gen.fs)
    sqi_ppg_res = estimate_sqi_and_hr(window.ppg, gen.fs)

    present_ecg = ecg_level < 4
    present_ppg = ppg_level < 4

    sqi_ecg = np.where(present_ecg, sqi_ecg_res.sqi, 0.0)
    sqi_ppg = np.where(present_ppg, sqi_ppg_res.sqi, 0.0)

    hr_est_ecg = np.where(present_ecg & np.isfinite(sqi_ecg_res.hr_est_bpm), sqi_ecg_res.hr_est_bpm, window.hr_true_bpm)
    hr_est_ppg = np.where(present_ppg & np.isfinite(sqi_ppg_res.hr_est_bpm), sqi_ppg_res.hr_est_bpm, window.hr_true_bpm)

    return PhysiologicalDataset(
        ecg=window.ecg, ppg=window.ppg, hr_true_bpm=window.hr_true_bpm,
        ecg_level=ecg_level, ppg_level=ppg_level, regime=regime,
        sqi_ecg=sqi_ecg, sqi_ppg=sqi_ppg,
        hr_est_ecg=hr_est_ecg, hr_est_ppg=hr_est_ppg,
        present_ecg=present_ecg, present_ppg=present_ppg, fs=gen.fs,
    )
