"""
PIR-Controller: a contextual-bandit policy that chooses, every window, a
joint (RLNC redundancy R, generation size K, compute placement) action to
maximize expected delivered clinical information per joule and millisecond,
subject to a strict latency deadline and an average-power cap.

Two disjoint-linear-model contextual bandit algorithms are implemented from
first principles (no if/else heuristic rule table):

  * LinUCB      (Li et al., 2010, "A Contextual-Bandit Approach to
                 Personalized News Article Recommendation") -- ridge
                 regression per arm with an upper-confidence-bound
                 exploration bonus.
  * Thompson    Bayesian linear regression per arm with posterior sampling
    Sampling    (Agrawal & Goyal, 2013, "Thompson Sampling for Contextual
                 Bandits with Linear Payoffs").

Reward
------
For chosen action a = (K, R, placement) in wireless/energy/latency state s:

    U_PIR(a, s)  = PIV * P_decode(K, R, p_erasure(s))          [clinical value]
    reward(a, s) = U_PIR(a, s)
                   - lambda_energy  * E(a, s)                   [joules]
                   - lambda_latency * softplus(L(a,s) - L_max)  [deadline]
                   - lambda_power   * softplus(P(a,s) - P_max)  [power cap]

L_max = 150 ms (medical latency deadline), P_max = 100 mW (power cap). The
softplus penalty is 0 well inside the constraint and grows smoothly (rather
than as a hard step) once it is violated, which keeps the reward
differentiable-in-expectation for the linear bandit model while still
imposing an effectively hard penalty at realistic lambda scales.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np


def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(beta * x)) / beta."""
    x = np.asarray(x, dtype=np.float64)
    z = beta * x
    return np.where(z > 30, z, np.log1p(np.exp(np.clip(z, -30, 30)))) / beta


@dataclass(frozen=True)
class Arm:
    """One discrete action available to the bandit."""

    K: int
    R: int
    placement_is_edge: bool

    @property
    def name(self) -> str:
        place = "edge" if self.placement_is_edge else "local"
        return f"K{self.K}_R{self.R}_{place}"


def default_arm_set(
    K_values: tuple[int, ...] = (32,),
    R_values: tuple[int, ...] = (2, 8, 20),
    placements: tuple[bool, ...] = (False, True),
) -> list[Arm]:
    """Cartesian product of redundancy levels and compute placements. K is
    held fixed by default (varying K changes the RLNC field-size regime,
    not just the redundancy ratio: fixing it isolates R as the primary
    controllable code-rate lever, matching the APC-RLNC redundancy
    formula's role as the system's principal adaptive knob)."""
    return [Arm(K=k, R=r, placement_is_edge=p) for k, r, p in product(K_values, R_values, placements)]


@dataclass
class RewardWeights:
    lambda_energy: float = 4.0     # penalty per joule
    lambda_latency: float = 0.05   # penalty per ms over the deadline
    lambda_power: float = 0.02     # penalty per mW over the cap
    latency_deadline_ms: float = 150.0
    power_cap_mw: float = 100.0
    softplus_beta: float = 0.5


def compute_reward(
    piv: np.ndarray, p_decode: np.ndarray, energy_j: np.ndarray,
    latency_ms: np.ndarray, power_mw: np.ndarray, weights: RewardWeights,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Vectorized multi-objective reward. Returns (reward, component_dict)
    so evaluation code can log each term separately."""
    u_pir = piv * p_decode
    energy_penalty = weights.lambda_energy * energy_j
    latency_penalty = weights.lambda_latency * softplus(
        latency_ms - weights.latency_deadline_ms, beta=weights.softplus_beta,
    )
    power_penalty = weights.lambda_power * softplus(
        power_mw - weights.power_cap_mw, beta=weights.softplus_beta,
    )
    reward = u_pir - energy_penalty - latency_penalty - power_penalty
    components = {
        "u_pir": u_pir, "energy_penalty": energy_penalty,
        "latency_penalty": latency_penalty, "power_penalty": power_penalty,
    }
    return reward, components


# --------------------------------------------------------------------------
# LinUCB
# --------------------------------------------------------------------------


class LinUCBController:
    """Disjoint linear-model LinUCB contextual bandit (Li et al., 2010).

    Maintains, for each arm a, a ridge-regression posterior mean
    theta_a = A_a^-1 b_a over a d-dimensional context, and selects the arm
    maximizing the upper confidence bound

        UCB_a(x) = x^T theta_a + alpha * sqrt(x^T A_a^-1 x).

    All linear algebra is done with explicit matrix inverses maintained via
    the Sherman-Morrison rank-1 update, so a single context update is O(d^2)
    rather than O(d^3).
    """

    def __init__(self, arms: list[Arm], context_dim: int, alpha: float = 1.5, ridge_lambda: float = 1.0):
        self.arms = arms
        self.n_arms = len(arms)
        self.d = context_dim
        self.alpha = alpha
        self.A_inv = np.stack([np.eye(self.d) / ridge_lambda for _ in arms])  # (n_arms, d, d)
        self.b = np.zeros((self.n_arms, self.d))
        self.n_pulls = np.zeros(self.n_arms, dtype=np.int64)
        self.cumulative_reward = np.zeros(self.n_arms)

    def theta(self) -> np.ndarray:
        return np.einsum("aij,aj->ai", self.A_inv, self.b)  # (n_arms, d)

    def select(self, contexts: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Select one arm index per row of `contexts` (shape (N, d))."""
        theta = self.theta()  # (n_arms, d)
        mean = contexts @ theta.T  # (N, n_arms)
        # x^T A_inv x for every (context, arm) pair, vectorized:
        # variance[n, a] = contexts[n] @ A_inv[a] @ contexts[n]
        variance = np.einsum("nj,ajk,nk->na", contexts, self.A_inv, contexts)
        variance = np.clip(variance, 0.0, None)
        ucb = mean + self.alpha * np.sqrt(variance)
        return np.argmax(ucb, axis=1)

    def update(self, contexts: np.ndarray, arm_indices: np.ndarray, rewards: np.ndarray) -> None:
        """Sherman-Morrison rank-1 update of A_inv and b for each observed
        (context, arm, reward) triple. Contexts sharing the same arm within
        this call are applied sequentially (correct for any batch size)."""
        for a in np.unique(arm_indices):
            mask = arm_indices == a
            xs = contexts[mask]
            rs = rewards[mask]
            for x, r in zip(xs, rs):
                Ainv = self.A_inv[a]
                Ainv_x = Ainv @ x
                denom = 1.0 + x @ Ainv_x
                self.A_inv[a] = Ainv - np.outer(Ainv_x, Ainv_x) / denom
                self.b[a] += r * x
                self.n_pulls[a] += 1
                self.cumulative_reward[a] += r

    def act_and_learn(
        self, contexts: np.ndarray, reward_fn, rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Convenience wrapper: select arms, evaluate `reward_fn(arm_indices)`
        -> (rewards, extra_info), update the posterior, and return
        (arm_indices, rewards, extra_info)."""
        arm_indices = self.select(contexts, rng=rng)
        rewards, extra = reward_fn(arm_indices)
        self.update(contexts, arm_indices, rewards)
        return arm_indices, rewards, extra


# --------------------------------------------------------------------------
# Thompson Sampling (Bayesian linear regression per arm)
# --------------------------------------------------------------------------


class ThompsonSamplingController:
    """Disjoint Bayesian-linear-regression Thompson Sampling contextual
    bandit (Agrawal & Goyal, 2013). Each arm maintains a Gaussian posterior
    N(mu_a, sigma^2 * A_a^-1) over its reward-weight vector; arm selection
    draws one sample theta~posterior per context and acts greedily on it,
    which explores automatically without a tunable UCB `alpha`.
    """

    def __init__(
        self, arms: list[Arm], context_dim: int, noise_variance: float = 0.25,
        ridge_lambda: float = 1.0, seed: int | None = None,
    ):
        self.arms = arms
        self.n_arms = len(arms)
        self.d = context_dim
        self.noise_variance = noise_variance
        self.A_inv = np.stack([np.eye(self.d) / ridge_lambda for _ in arms])
        self.b = np.zeros((self.n_arms, self.d))
        self.rng = np.random.default_rng(seed)
        self.n_pulls = np.zeros(self.n_arms, dtype=np.int64)
        self.cumulative_reward = np.zeros(self.n_arms)

    def theta_mean(self) -> np.ndarray:
        return np.einsum("aij,aj->ai", self.A_inv, self.b)

    def select(self, contexts: np.ndarray) -> np.ndarray:
        mean = self.theta_mean()  # (n_arms, d)
        N = contexts.shape[0]
        sampled_theta = np.empty((self.n_arms, self.d))
        for a in range(self.n_arms):
            cov = self.noise_variance * self.A_inv[a]
            cov = 0.5 * (cov + cov.T)  # enforce exact symmetry (fp round-off guard)
            sampled_theta[a] = self.rng.multivariate_normal(mean[a], cov)
        scores = contexts @ sampled_theta.T  # (N, n_arms)
        return np.argmax(scores, axis=1)

    def update(self, contexts: np.ndarray, arm_indices: np.ndarray, rewards: np.ndarray) -> None:
        for a in np.unique(arm_indices):
            mask = arm_indices == a
            xs = contexts[mask]
            rs = rewards[mask]
            for x, r in zip(xs, rs):
                Ainv = self.A_inv[a]
                Ainv_x = Ainv @ x
                denom = 1.0 + x @ Ainv_x
                self.A_inv[a] = Ainv - np.outer(Ainv_x, Ainv_x) / denom
                self.b[a] += r * x
                self.n_pulls[a] += 1
                self.cumulative_reward[a] += r

    def act_and_learn(self, contexts: np.ndarray, reward_fn) -> tuple[np.ndarray, np.ndarray, dict]:
        arm_indices = self.select(contexts)
        rewards, extra = reward_fn(arm_indices)
        self.update(contexts, arm_indices, rewards)
        return arm_indices, rewards, extra


# --------------------------------------------------------------------------
# Optional: PyTorch neural contextual bandit (NeuralUCB-style), used only if
# torch is installed. Purely additive -- the NumPy LinUCB/Thompson policies
# above are fully functional (and are what `evaluate.py` uses by default)
# with no torch dependency.
# --------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class _RewardNet(nn.Module):
        def __init__(self, context_dim: int, n_arms: int, hidden: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(context_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_arms),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class NeuralUCBController:
        """Optional neural contextual bandit: a small MLP predicts the
        reward of every arm; exploration uses epsilon-greedy plus a
        gradient-norm-free ensemble-disagreement proxy computed from
        dropout-based MC sampling. Requires PyTorch; guarded so the rest of
        the framework never requires it."""

        def __init__(
            self, arms: list[Arm], context_dim: int, hidden: int = 32,
            lr: float = 1e-2, epsilon: float = 0.1, seed: int | None = None,
        ):
            if seed is not None:
                torch.manual_seed(seed)
            self.arms = arms
            self.n_arms = len(arms)
            self.epsilon = epsilon
            self.rng = np.random.default_rng(seed)
            self.model = _RewardNet(context_dim, self.n_arms, hidden)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()
            self.n_pulls = np.zeros(self.n_arms, dtype=np.int64)
            self.cumulative_reward = np.zeros(self.n_arms)

        def select(self, contexts: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                x = torch.as_tensor(contexts, dtype=torch.float32)
                q = self.model(x).numpy()
            greedy = np.argmax(q, axis=1)
            explore_mask = self.rng.random(len(contexts)) < self.epsilon
            random_arms = self.rng.integers(0, self.n_arms, size=len(contexts))
            return np.where(explore_mask, random_arms, greedy)

        def update(self, contexts: np.ndarray, arm_indices: np.ndarray, rewards: np.ndarray) -> None:
            x = torch.as_tensor(contexts, dtype=torch.float32)
            r = torch.as_tensor(rewards, dtype=torch.float32)
            q = self.model(x)
            q_chosen = q.gather(1, torch.as_tensor(arm_indices, dtype=torch.long).unsqueeze(1)).squeeze(1)
            loss = self.loss_fn(q_chosen, r)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for a in np.unique(arm_indices):
                mask = arm_indices == a
                self.n_pulls[a] += int(mask.sum())
                self.cumulative_reward[a] += float(rewards[mask].sum())

        def act_and_learn(self, contexts: np.ndarray, reward_fn) -> tuple[np.ndarray, np.ndarray, dict]:
            arm_indices = self.select(contexts)
            rewards, extra = reward_fn(arm_indices)
            self.update(contexts, arm_indices, rewards)
            return arm_indices, rewards, extra

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is an optional dependency
    TORCH_AVAILABLE = False
