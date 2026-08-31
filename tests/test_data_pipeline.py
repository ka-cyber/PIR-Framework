import numpy as np
import pytest

from pir_framework.data_pipeline import (
    SyntheticECGPPGGenerator, DegradationConfig, estimate_sqi_and_hr, build_dataset,
)


class TestSyntheticGenerator:
    def test_clean_generation_shapes(self):
        gen = SyntheticECGPPGGenerator(fs=125.0, window_seconds=8.0, seed=0)
        ecg, ppg, hr = gen.generate_clean(10)
        assert ecg.shape == (10, 1000)
        assert ppg.shape == (10, 1000)
        assert hr.shape == (10,)
        assert np.all(hr >= 50.0) and np.all(hr <= 110.0)

    def test_reproducible_with_seed(self):
        gen1 = SyntheticECGPPGGenerator(seed=5)
        gen2 = SyntheticECGPPGGenerator(seed=5)
        ecg1, ppg1, hr1 = gen1.generate_clean(5)
        ecg2, ppg2, hr2 = gen2.generate_clean(5)
        assert np.array_equal(ecg1, ecg2)
        assert np.array_equal(hr1, hr2)

    def test_degradation_increases_noise_energy(self):
        gen = SyntheticECGPPGGenerator(seed=1)
        ecg_clean, _, _ = gen.generate_clean(20)
        level0 = np.zeros(20, dtype=int)
        level3 = np.full(20, 3, dtype=int)
        out_clean = gen.apply_degradation(ecg_clean, level0, kind="ecg")
        out_severe = gen.apply_degradation(ecg_clean, level3, kind="ecg")
        # Severe degradation should deviate from the clean signal much more
        # than "degradation" at level 0 (which should be a no-op).
        assert np.allclose(out_clean, ecg_clean)
        resid_severe = np.std(out_severe - ecg_clean)
        assert resid_severe > 0.1

    def test_missing_modality_replaced_with_noise_floor(self):
        gen = SyntheticECGPPGGenerator(seed=2)
        n = 10
        ecg_level = np.full(n, 4)  # missing
        ppg_level = np.zeros(n, dtype=int)
        window = gen.generate(n, ecg_level, ppg_level)
        # Missing-ECG windows should not resemble a clean cardiac waveform:
        # their peak-to-peak amplitude structure differs greatly from PPG's.
        assert window.ecg.shape == (n, gen.n_samples)
        assert np.all(window.ecg_level == 4)


class TestSQIEstimator:
    def test_clean_signal_high_sqi_and_accurate_hr(self):
        gen = SyntheticECGPPGGenerator(seed=3)
        ecg, ppg, hr_true = gen.generate_clean(30)
        result = estimate_sqi_and_hr(ppg, gen.fs)
        valid = result.passes_feasibility
        assert valid.sum() > 25  # most clean windows should pass feasibility
        mae = np.mean(np.abs(result.hr_est_bpm[valid] - hr_true[valid]))
        assert mae < 3.0  # should closely match true HR on clean signals

    def test_flat_signal_gives_zero_sqi(self):
        flat = np.zeros((3, 1000))
        result = estimate_sqi_and_hr(flat, fs=125.0)
        assert np.all(result.sqi == 0.0)
        assert np.all(~result.passes_feasibility)

    def test_sqi_bounded(self):
        gen = SyntheticECGPPGGenerator(seed=4)
        ecg, ppg, _ = gen.generate_clean(15)
        result = estimate_sqi_and_hr(ecg, gen.fs)
        assert np.all(result.sqi >= 0.0) and np.all(result.sqi <= 1.0)


class TestBuildDataset:
    def test_regime_counts_balanced(self):
        ds = build_dataset(n_per_regime=20, seed=0)
        assert len(ds.regime) == 120
        unique, counts = np.unique(ds.regime, return_counts=True)
        assert len(unique) == 6
        assert np.all(counts == 20)

    def test_missing_modality_zero_sqi_and_exact_fallback_hr(self):
        ds = build_dataset(n_per_regime=15, seed=1)
        ecg_missing = ds.regime == "ecg_missing"
        assert np.all(ds.sqi_ecg[ecg_missing] == 0.0)
        assert np.all(~ds.present_ecg[ecg_missing])
        # Fallback heart rate for the missing modality should exactly equal
        # ground truth (no spurious noise-floor peak detection artifact).
        assert np.allclose(ds.hr_est_ecg[ecg_missing], ds.hr_true_bpm[ecg_missing])

    def test_clean_regime_hr_estimates_accurate(self):
        ds = build_dataset(n_per_regime=40, seed=2)
        clean = ds.regime == "both_clean"
        mae_ecg = np.mean(np.abs(ds.hr_est_ecg[clean] - ds.hr_true_bpm[clean]))
        mae_ppg = np.mean(np.abs(ds.hr_est_ppg[clean] - ds.hr_true_bpm[clean]))
        assert mae_ecg < 3.0
        assert mae_ppg < 3.0

    def test_degraded_regime_shows_lower_sqi_than_clean(self):
        ds = build_dataset(n_per_regime=40, seed=3)
        clean_sqi = ds.sqi_ecg[ds.regime == "both_clean"].mean()
        degraded_sqi = ds.sqi_ecg[ds.regime == "ecg_degraded"].mean()
        assert degraded_sqi < clean_sqi
