import numpy as np
import pytest

from pir_framework.pir_controller import (
    Arm, default_arm_set, LinUCBController, ThompsonSamplingController,
    RewardWeights, compute_reward, softplus,
)


class TestArms:
    def test_default_arm_set_cartesian_product(self):
        arms = default_arm_set(K_values=(32,), R_values=(2, 8), placements=(False, True))
        assert len(arms) == 4
        names = {a.name for a in arms}
        assert "K32_R2_local" in names
        assert "K32_R8_edge" in names


class TestSoftplus:
    def test_softplus_zero_at_zero(self):
        assert softplus(np.array([0.0]))[0] == pytest.approx(np.log(2), abs=1e-6)

    def test_softplus_nonnegative(self):
        x = np.linspace(-50, 50, 200)
        assert np.all(softplus(x) >= 0.0)

    def test_softplus_approaches_identity_for_large_positive_x(self):
        x = np.array([50.0])
        assert softplus(x)[0] == pytest.approx(50.0, abs=1e-3)

    def test_softplus_approaches_zero_for_large_negative_x(self):
        x = np.array([-50.0])
        assert softplus(x)[0] < 1e-6


class TestRewardFunction:
    def test_reward_decreases_with_energy(self):
        w = RewardWeights()
        piv, p_decode, latency, power = np.array([0.8]), np.array([0.9]), np.array([50.0]), np.array([10.0])
        r_low, _ = compute_reward(piv, p_decode, np.array([0.01]), latency, power, w)
        r_high, _ = compute_reward(piv, p_decode, np.array([0.5]), latency, power, w)
        assert r_low > r_high

    def test_reward_penalizes_deadline_violation(self):
        w = RewardWeights(latency_deadline_ms=150.0)
        piv, p_decode, energy, power = np.array([0.8]), np.array([0.9]), np.array([0.05]), np.array([10.0])
        r_ok, comp_ok = compute_reward(piv, p_decode, energy, np.array([100.0]), power, w)
        r_late, comp_late = compute_reward(piv, p_decode, energy, np.array([300.0]), power, w)
        assert r_ok > r_late
        assert comp_late["latency_penalty"][0] > comp_ok["latency_penalty"][0]

    def test_reward_penalizes_power_cap_violation(self):
        w = RewardWeights(power_cap_mw=100.0)
        piv, p_decode, energy, latency = np.array([0.8]), np.array([0.9]), np.array([0.05]), np.array([50.0])
        r_ok, _ = compute_reward(piv, p_decode, energy, latency, np.array([50.0]), w)
        r_over, _ = compute_reward(piv, p_decode, energy, latency, np.array([500.0]), w)
        assert r_ok > r_over

    def test_reward_increases_with_decode_probability(self):
        w = RewardWeights()
        piv, energy, latency, power = np.array([0.8]), np.array([0.05]), np.array([50.0]), np.array([10.0])
        r_low, _ = compute_reward(piv, np.array([0.1]), energy, latency, power, w)
        r_high, _ = compute_reward(piv, np.array([0.99]), energy, latency, power, w)
        assert r_high > r_low


class TestLinUCB:
    def test_select_returns_valid_arm_indices(self):
        arms = default_arm_set()
        ctrl = LinUCBController(arms, context_dim=10, alpha=1.0)
        ctx = np.random.default_rng(0).normal(size=(20, 10))
        chosen = ctrl.select(ctx)
        assert chosen.shape == (20,)
        assert np.all((chosen >= 0) & (chosen < len(arms)))

    def test_update_changes_theta(self):
        arms = default_arm_set()
        ctrl = LinUCBController(arms, context_dim=10, alpha=1.0)
        theta_before = ctrl.theta().copy()
        ctx = np.ones((5, 10))
        arm_idx = np.zeros(5, dtype=np.int64)
        rewards = np.ones(5) * 3.0
        ctrl.update(ctx, arm_idx, rewards)
        theta_after = ctrl.theta()
        assert not np.allclose(theta_before[0], theta_after[0])

    def test_pull_counts_accumulate(self):
        arms = default_arm_set()
        ctrl = LinUCBController(arms, context_dim=10, alpha=1.0)
        ctx = np.zeros((7, 10))
        arm_idx = np.full(7, 2, dtype=np.int64)
        ctrl.update(ctx, arm_idx, np.ones(7))
        assert ctrl.n_pulls[2] == 7
        assert ctrl.n_pulls.sum() == 7

    def test_learns_to_prefer_higher_reward_arm(self):
        # Two arms, arm 1 always yields higher reward regardless of context;
        # after many rounds LinUCB should pull it far more often. A bias
        # (constant 1.0) column is required in the context so a purely
        # context-independent reward difference is representable by the
        # linear model at all (a mean-zero random context alone cannot
        # encode a constant preference, since E[theta . x] = 0 for any
        # theta when E[x] = 0).
        arms = [Arm(K=32, R=2, placement_is_edge=False), Arm(K=32, R=20, placement_is_edge=False)]
        ctrl = LinUCBController(arms, context_dim=5, alpha=0.3, ridge_lambda=1.0)
        rng = np.random.default_rng(0)
        for _ in range(300):
            ctx = np.concatenate([rng.normal(size=(4, 4)), np.ones((4, 1))], axis=1)
            chosen = ctrl.select(ctx)
            rewards = np.where(chosen == 1, 1.0, 0.0) + rng.normal(0, 0.01, size=4)
            ctrl.update(ctx, chosen, rewards)
        assert ctrl.n_pulls[1] > ctrl.n_pulls[0]

    def test_act_and_learn_convenience(self):
        arms = default_arm_set()
        ctrl = LinUCBController(arms, context_dim=10, alpha=1.0)
        ctx = np.random.default_rng(0).normal(size=(3, 10))

        def reward_fn(arm_indices):
            return np.ones(len(arm_indices)), {}

        arm_idx, rewards, extra = ctrl.act_and_learn(ctx, reward_fn)
        assert arm_idx.shape == (3,)
        assert np.allclose(rewards, 1.0)


class TestThompsonSampling:
    def test_select_returns_valid_arm_indices(self):
        arms = default_arm_set()
        ctrl = ThompsonSamplingController(arms, context_dim=10, seed=0)
        ctx = np.random.default_rng(0).normal(size=(15, 10))
        chosen = ctrl.select(ctx)
        assert chosen.shape == (15,)
        assert np.all((chosen >= 0) & (chosen < len(arms)))

    def test_learns_to_prefer_higher_reward_arm(self):
        arms = [Arm(K=32, R=2, placement_is_edge=False), Arm(K=32, R=20, placement_is_edge=False)]
        ctrl = ThompsonSamplingController(arms, context_dim=5, noise_variance=0.05, seed=1)
        rng = np.random.default_rng(1)
        for _ in range(300):
            ctx = np.concatenate([rng.normal(size=(4, 4)), np.ones((4, 1))], axis=1)
            chosen = ctrl.select(ctx)
            rewards = np.where(chosen == 1, 1.0, 0.0) + rng.normal(0, 0.01, size=4)
            ctrl.update(ctx, chosen, rewards)
        assert ctrl.n_pulls[1] > ctrl.n_pulls[0]

    def test_covariance_matrix_stays_symmetric_psd(self):
        arms = default_arm_set()
        ctrl = ThompsonSamplingController(arms, context_dim=6, seed=2)
        rng = np.random.default_rng(2)
        for _ in range(50):
            ctx = rng.normal(size=(3, 6))
            chosen = ctrl.select(ctx)
            ctrl.update(ctx, chosen, rng.normal(size=3))
        for a in range(len(arms)):
            cov = ctrl.A_inv[a]
            assert np.allclose(cov, cov.T, atol=1e-8)
            eigvals = np.linalg.eigvalsh(cov)
            assert np.all(eigvals > -1e-8)
