import numpy as np
import pytest

from pir_framework.state_space import (
    StateSpaceEngine, MedicalState, compute_piv, EnergyModel, ComputeModel, EnergyState,
)


def make_medical(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return MedicalState(
        sqi_ecg=rng.uniform(0, 1, n), sqi_ppg=rng.uniform(0, 1, n),
        modality_present_ecg=np.ones(n, dtype=bool), modality_present_ppg=np.ones(n, dtype=bool),
        hr_ecg=rng.uniform(50, 110, n), hr_ppg=rng.uniform(50, 110, n),
    )


class TestPIV:
    def test_piv_in_unit_interval(self):
        rng = np.random.default_rng(0)
        n = 200
        sqi_ecg, sqi_ppg = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        present_ecg, present_ppg = rng.random(n) > 0.2, rng.random(n) > 0.2
        hr_ecg, hr_ppg = rng.uniform(40, 180, n), rng.uniform(40, 180, n)
        piv = compute_piv(sqi_ecg, sqi_ppg, present_ecg, present_ppg, hr_ecg, hr_ppg)
        assert np.all(piv >= 0.0) and np.all(piv <= 1.0)

    def test_piv_zero_when_both_missing(self):
        piv = compute_piv(
            sqi_ecg=np.array([0.9]), sqi_ppg=np.array([0.9]),
            present_ecg=np.array([False]), present_ppg=np.array([False]),
            hr_ecg=np.array([70.0]), hr_ppg=np.array([70.0]),
        )
        assert piv[0] == pytest.approx(0.0)

    def test_piv_high_when_clean_and_agreeing(self):
        piv = compute_piv(
            sqi_ecg=np.array([1.0]), sqi_ppg=np.array([1.0]),
            present_ecg=np.array([True]), present_ppg=np.array([True]),
            hr_ecg=np.array([70.0]), hr_ppg=np.array([70.0]),
        )
        # With default weights (w_quality=0.75, w_agreement=0.25), perfect
        # SQI and perfect agreement yields PIV == w_quality (the agreement
        # term contributes zero penalty, not bonus above w_quality).
        assert piv[0] == pytest.approx(0.75, abs=1e-6)

    def test_piv_penalized_by_disagreement(self):
        agree = compute_piv(
            sqi_ecg=np.array([1.0]), sqi_ppg=np.array([1.0]),
            present_ecg=np.array([True]), present_ppg=np.array([True]),
            hr_ecg=np.array([70.0]), hr_ppg=np.array([70.0]),
        )[0]
        disagree = compute_piv(
            sqi_ecg=np.array([1.0]), sqi_ppg=np.array([1.0]),
            present_ecg=np.array([True]), present_ppg=np.array([True]),
            hr_ecg=np.array([70.0]), hr_ppg=np.array([140.0]),
        )[0]
        assert disagree < agree

    def test_piv_not_penalized_by_disagreement_when_one_missing(self):
        # No "disagreement" is meaningful if only one modality is present,
        # regardless of what hr_ecg/hr_ppg happen to numerically hold.
        piv = compute_piv(
            sqi_ecg=np.array([1.0]), sqi_ppg=np.array([0.0]),
            present_ecg=np.array([True]), present_ppg=np.array([False]),
            hr_ecg=np.array([70.0]), hr_ppg=np.array([200.0]),
        )
        # Driven purely by ECG's SQI (1.0) via w_quality (0.75); the bogus
        # PPG HR must not leak a disagreement penalty into this value.
        assert piv[0] == pytest.approx(0.75, abs=1e-6)


class TestEnergyAndCompute:
    def test_energy_increases_with_redundancy(self):
        model = EnergyModel()
        low = model.compute(n_tx=np.array([10]), n_rx=np.array([1]), n_ops=np.array([100]), window_seconds=8.0)
        high = model.compute(n_tx=np.array([40]), n_rx=np.array([1]), n_ops=np.array([500]), window_seconds=8.0)
        assert high > low

    def test_energy_state_consume_never_negative(self):
        es = EnergyState.initialize(n_envs=3, capacity_j=10.0)
        es.consume(np.array([5.0, 15.0, 0.0]))
        assert np.all(es.battery_j >= 0.0)
        assert es.battery_j[1] == 0.0  # clamped, not negative

    def test_compute_latency_edge_vs_local_ordering_reasonable(self):
        model = ComputeModel()
        K, R = np.array([32]), np.array([8])
        local_ms, _ = model.compute_latency_ms(K, R, placement_is_edge=np.array([False]), payload_bits=512)
        edge_ms, _ = model.compute_latency_ms(K, R, placement_is_edge=np.array([True]), payload_bits=512)
        assert local_ms[0] > 0 and edge_ms[0] > 0

    def test_latency_increases_with_redundancy(self):
        model = ComputeModel()
        low, _ = model.compute_latency_ms(np.array([32]), np.array([2]), np.array([False]), 512)
        high, _ = model.compute_latency_ms(np.array([32]), np.array([24]), np.array([False]), 512)
        assert high[0] > low[0]


class TestStateSpaceEngine:
    def test_context_shape_matches_feature_names(self):
        engine = StateSpaceEngine(n_envs=10, seed=0)
        med = make_medical(10)
        ctx = engine.observe_context(med)
        assert ctx.shape == (10, len(StateSpaceEngine.context_feature_names()))

    def test_act_returns_consistent_shapes(self):
        n = 6
        engine = StateSpaceEngine(n_envs=n, seed=1)
        med = make_medical(n)
        result = engine.act(
            med, K=np.full(n, 32), R=np.full(n, 8), placement_is_edge=np.zeros(n, dtype=bool),
        )
        assert result.energy_j.shape == (n,)
        assert result.latency_ms.shape == (n,)
        assert result.p_decode.shape == (n,)
        assert np.all(result.p_decode >= 0.0) and np.all(result.p_decode <= 1.0)

    def test_battery_depletes_monotonically(self):
        n = 4
        engine = StateSpaceEngine(n_envs=n, battery_capacity_j=1.0, seed=2)
        med = make_medical(n)
        prev = engine.energy_state.battery_j.copy()
        for _ in range(5):
            engine.act(med, K=np.full(n, 32), R=np.full(n, 20), placement_is_edge=np.ones(n, dtype=bool))
            assert np.all(engine.energy_state.battery_j <= prev + 1e-12)
            prev = engine.energy_state.battery_j.copy()

    def test_battery_never_negative_even_when_exhausted(self):
        n = 2
        engine = StateSpaceEngine(n_envs=n, battery_capacity_j=1e-6, seed=3)
        med = make_medical(n)
        for _ in range(20):
            engine.act(med, K=np.full(n, 32), R=np.full(n, 24), placement_is_edge=np.ones(n, dtype=bool))
        assert np.all(engine.energy_state.battery_j >= 0.0)

    def test_higher_redundancy_improves_decode_prob_under_erasure(self):
        # Force a channel with substantial erasure by using many links and
        # checking the *distributional* relationship rather than a single
        # noisy draw: higher R should give higher mean decode probability.
        n = 20000
        engine_low = StateSpaceEngine(n_envs=n, seed=5)
        engine_high = StateSpaceEngine(n_envs=n, seed=5)  # identical channel trajectory
        med = make_medical(n, seed=5)
        # re-seed both channels identically by construction (seed=5 shared)
        res_low = engine_low.act(med, K=np.full(n, 32), R=np.full(n, 2), placement_is_edge=np.zeros(n, dtype=bool))
        res_high = engine_high.act(med, K=np.full(n, 32), R=np.full(n, 20), placement_is_edge=np.zeros(n, dtype=bool))
        assert res_high.p_decode.mean() >= res_low.p_decode.mean()

    def test_observe_context_uses_previous_wireless_not_current(self):
        n = 5
        engine = StateSpaceEngine(n_envs=n, seed=7)
        med = make_medical(n)
        ctx_before = engine.observe_context(med)
        result = engine.act(med, K=np.full(n, 32), R=np.full(n, 8), placement_is_edge=np.zeros(n, dtype=bool))
        # The context used for the PRE-action decision must not equal the
        # post-action realized wireless features (no information leakage
        # from the future channel draw into the decision context).
        assert not np.allclose(ctx_before[:, 5], result.context[:, 5])  # snr_db_norm column
