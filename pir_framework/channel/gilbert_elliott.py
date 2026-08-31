"""
Gilbert-Elliott two-state Markov erasure channel, extended with an
SNR-linked bit-error-rate (BER) model so that "instantaneous SNR (dB)" and
"burst-erasure probability" are two views of the *same* underlying physical
state rather than two independently-specified numbers.

Model
-----
Each of N parallel links/nodes independently occupies a hidden state
s_i(t) in {GOOD, BAD}, evolving as a 2-state Markov chain with transition
probabilities P(GOOD->BAD) = p_gb, P(BAD->GOOD) = p_bg (matches the APC-RLNC
channel model). Conditioned on state, instantaneous SNR is drawn from a
state-dependent Gaussian:

    SNR_dB(t) | GOOD ~ N(mu_good, sigma_good^2)
    SNR_dB(t) | BAD  ~ N(mu_bad,  sigma_bad^2)

The per-packet bit-error rate is then obtained from SNR via the standard
BPSK-over-AWGN closed form BER = Q(sqrt(2 * SNR_linear)), and the
packet-erasure probability from BER via the union bound over an
L-bit payload: p_erasure = 1 - (1 - BER)^L.

This keeps the model consistent with both the *bursty* (state-persistence)
character of Gilbert-Elliott channels used in APC-RLNC, and a physically
interpretable SNR variable that the PIR-Controller can condition on
directly, rather than treating "SNR" and "erasure probability" as two
disconnected numbers as a purely rule-based system would.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

GOOD, BAD = 0, 1


def snr_to_ber_bpsk(snr_db: np.ndarray) -> np.ndarray:
    """BPSK-over-AWGN bit-error rate: BER = Q(sqrt(2 * 10^(SNR_dB/10)))."""
    snr_db = np.asarray(snr_db, dtype=np.float64)
    snr_linear = 10.0 ** (snr_db / 10.0)
    return norm.sf(np.sqrt(2.0 * snr_linear))  # Q(x) = 1 - Phi(x) = norm.sf(x)


def ber_to_packet_erasure(ber: np.ndarray, payload_bits: int) -> np.ndarray:
    """Packet erasure probability from bit-error rate via the union bound:
    a payload of `payload_bits` bits is erased if >= 1 bit is flipped."""
    ber = np.clip(np.asarray(ber, dtype=np.float64), 0.0, 1.0)
    return 1.0 - (1.0 - ber) ** payload_bits


@dataclass
class GilbertElliottChannel:
    """Vectorized N-link Gilbert-Elliott channel with an SNR-linked BER model.

    Parameters
    ----------
    n_links : number of independent, parallel links simulated.
    p_gb, p_bg : GOOD->BAD and BAD->GOOD transition probabilities per step.
    mu_good_db, sigma_good_db : mean/std of SNR (dB) while in the GOOD state.
    mu_bad_db, sigma_bad_db   : mean/std of SNR (dB) while in the BAD state.
    payload_bits : payload size used to map BER -> packet erasure probability.
    """

    n_links: int
    p_gb: float = 0.03
    p_bg: float = 0.08
    mu_good_db: float = 18.0
    sigma_good_db: float = 1.5
    mu_bad_db: float = 7.0
    sigma_bad_db: float = 1.0
    payload_bits: int = 8 * 64  # 64-byte payload, matching RLNC packets
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)
    state: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        # Initialize each link's state from the chain's stationary distribution.
        pi_bad = self.p_gb / (self.p_gb + self.p_bg + 1e-12)
        self.state = (self.rng.random(self.n_links) < pi_bad).astype(np.int8)  # 0=GOOD,1=BAD

    def stationary_bad_prob(self) -> float:
        return self.p_gb / (self.p_gb + self.p_bg + 1e-12)

    def step(self) -> dict[str, np.ndarray]:
        """Advance every link one Markov step and draw this step's SNR,
        BER, and packet-erasure probability.

        Returns a dict with keys: state, snr_db, ber, p_erasure (each shape (n_links,)).
        """
        u = self.rng.random(self.n_links)
        transition_prob = np.where(self.state == GOOD, self.p_gb, self.p_bg)
        flip = u < transition_prob
        self.state = np.where(flip, 1 - self.state, self.state).astype(np.int8)

        mu = np.where(self.state == GOOD, self.mu_good_db, self.mu_bad_db)
        sigma = np.where(self.state == GOOD, self.sigma_good_db, self.sigma_bad_db)
        snr_db = self.rng.normal(mu, sigma)

        ber = snr_to_ber_bpsk(snr_db)
        p_erasure = ber_to_packet_erasure(ber, self.payload_bits)
        return {
            "state": self.state.copy(),
            "snr_db": snr_db,
            "ber": ber,
            "p_erasure": p_erasure,
        }

    def simulate(self, n_steps: int) -> dict[str, np.ndarray]:
        """Run `n_steps` steps, returning stacked arrays of shape (n_steps, n_links)."""
        out = {"state": [], "snr_db": [], "ber": [], "p_erasure": []}
        for _ in range(n_steps):
            step_out = self.step()
            for k, v in step_out.items():
                out[k].append(v)
        return {k: np.stack(v, axis=0) for k, v in out.items()}
