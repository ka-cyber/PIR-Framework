import numpy as np
import pytest

from pir_framework.evaluate import (
    SimulationConfig, run_single_seed, fuse_hr_estimate, slice_medical, simulate_policy,
)
from pir_framework.data_pipeline import build_dataset
from pir_framework.state_space import MedicalState


class TestFuseHR:
    def test_fused_hr_matches_ground_truth_when_clean(self):
        ds = build_dataset(n_per_regime=20, seed=0)
        fused = fuse_hr_estimate(ds)
        clean = ds.regime == "both_clean"
        mae = np.mean(np.abs(fused[clean] - ds.hr_true_bpm[clean]))
        assert mae < 3.0

    def test_fused_hr_finite_everywhere(self):
        ds = build_dataset(n_per_regime=15, seed=1)
        fused = fuse_hr_estimate(ds)
        assert np.all(np.isfinite(fused))


class TestSliceMedical:
    def test_slice_returns_length_one_state(self):
        ds = build_dataset(n_per_regime=5, seed=0)
        medical = MedicalState(
            sqi_ecg=ds.sqi_ecg, sqi_ppg=ds.sqi_ppg,
            modality_present_ecg=ds.present_ecg, modality_present_ppg=ds.present_ppg,
            hr_ecg=ds.hr_est_ecg, hr_ppg=ds.hr_est_ppg,
        )
        sliced = slice_medical(medical, 3)
        assert sliced.sqi_ecg.shape == (1,)
        assert sliced.piv.shape == (1,)


class TestRunSingleSeedSmall:
    """Small-scale (fast) end-to-end run to validate the full pipeline
    wiring without the cost of a full-size production run."""

    @pytest.fixture(scope="class")
    def result(self):
        config = SimulationConfig(n_per_regime=10, seed=0, outdir="unused")
        return run_single_seed(config, verbose=False)

    def test_all_five_policies_present(self, result):
        names = {p.name for p in result["policies"]}
        assert len(names) == 5
        assert "Fixed-Rate" in names
        assert "Heuristic-Rule" in names
        assert any("PIR-Contextual-Bandit" in n for n in names)

    def test_overall_metrics_one_row_per_policy(self, result):
        df = result["df_overall"]
        assert len(df) == 5
        for col in ["hr_mae_bpm", "pdr", "mean_energy_j", "mean_reward"]:
            assert col in df.columns
            assert df[col].notna().all()

    def test_steady_state_metrics_present(self, result):
        df = result["df_overall_steady"]
        assert len(df) == 5
        assert (df["n_windows"] > 0).all()

    def test_pdr_and_deadline_in_valid_range(self, result):
        df = result["df_overall"]
        assert (df["pdr"] >= 0.0).all() and (df["pdr"] <= 1.0).all()
        assert (df["deadline_success_rate"] >= 0.0).all()
        assert (df["deadline_success_rate"] <= 1.0).all()

    def test_bandit_arm_pulls_sum_to_n_windows(self, result):
        bandit_label = result["bandit_label"]
        pulls = result["arm_pull_counts"][bandit_label]
        assert pulls.sum() == result["n_total"]

    def test_reward_traces_length_matches_n_total(self, result):
        for name, trace in result["reward_traces_arrival_order"].items():
            assert trace.shape == (result["n_total"],)

    def test_non_adaptive_policies_use_single_fixed_arm(self, result):
        df_log = result["per_window_logs"]["Fixed-Rate"]
        assert df_log["arm"].nunique() == 1

    def test_bandit_explores_multiple_arms(self, result):
        df_log = result["per_window_logs"][result["bandit_label"]]
        assert df_log["arm"].nunique() > 1

    def test_reproducibility_same_seed(self):
        c1 = SimulationConfig(n_per_regime=8, seed=42, outdir="unused")
        c2 = SimulationConfig(n_per_regime=8, seed=42, outdir="unused")
        r1 = run_single_seed(c1, verbose=False)
        r2 = run_single_seed(c2, verbose=False)
        df1, df2 = r1["df_overall"], r2["df_overall"]
        assert np.allclose(df1["hr_mae_bpm"].to_numpy(), df2["hr_mae_bpm"].to_numpy())
        assert np.allclose(df1["mean_reward"].to_numpy(), df2["mean_reward"].to_numpy())

    def test_thompson_bandit_variant_runs(self):
        config = SimulationConfig(n_per_regime=8, seed=0, outdir="unused", bandit_type="thompson")
        result = run_single_seed(config, verbose=False)
        assert "Thompson" in result["bandit_label"]
        assert len(result["df_overall"]) == 5
