"""
Random Linear Network Coding (RLNC) over GF(256): encoder, Gaussian-elimination
decoder, and the closed-form decoding-probability model used by the
PIR-Controller's reward/latency estimates (mirrors APC-RLNC Eq. (3)).

A "generation" of K source packets is encoded into K + R coded packets by
transmitting random linear combinations of the source packets. A receiver
that collects any K linearly-independent coded packets (out of the K + R
sent, each independently erased with probability p) can recover the full
generation via Gaussian elimination over GF(256).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .gf256 import GF256, gf_inv, gf_mul, gf_add


@dataclass
class RLNCEncoder:
    """Encodes a generation of K source packets (each of `payload_bytes` bytes)
    into K + R coded packets using random GF(256) coding coefficients."""

    K: int
    R: int
    payload_bytes: int
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()

    @property
    def n_coded(self) -> int:
        return self.K + self.R

    def encode(self, source_packets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        source_packets : ndarray, shape (K, payload_bytes), dtype uint8

        Returns
        -------
        coding_vectors : ndarray, shape (K + R, K), dtype uint8
        coded_packets  : ndarray, shape (K + R, payload_bytes), dtype uint8
        """
        assert source_packets.shape == (self.K, self.payload_bytes)
        coding_vectors = GF256.random_nonzero_matrix(self.n_coded, self.K, self.rng)
        coded_packets = GF256.matmul(coding_vectors, source_packets.astype(np.uint8))
        return coding_vectors, coded_packets


class RLNCDecoder:
    """Incremental Gauss-Jordan RLNC decoder over GF(256)."""

    def __init__(self, K: int, payload_bytes: int) -> None:
        self.K = K
        self.payload_bytes = payload_bytes
        self._coeff_rows: list[np.ndarray] = []
        self._payload_rows: list[np.ndarray] = []

    @property
    def rank(self) -> int:
        return len(self._coeff_rows)

    @property
    def is_decoded(self) -> bool:
        return self.rank >= self.K

    def receive(self, coding_vector: np.ndarray, coded_payload: np.ndarray) -> bool:
        """Feed one received coded packet. Returns True if it increased rank
        (i.e. was linearly independent of previously-seen packets)."""
        v = coding_vector.astype(np.int64).copy()
        p = coded_payload.astype(np.uint8).copy()

        for cv, cp in zip(self._coeff_rows, self._payload_rows):
            pivot_col = int(np.argmax(cv != 0))
            if v[pivot_col] != 0:
                factor = int(v[pivot_col])
                v = gf_add(v, gf_mul(factor, cv))
                p = gf_add(p, gf_mul(factor, cp))

        if not np.any(v):
            return False  # linearly dependent -> does not raise rank

        pivot_col = int(np.argmax(v != 0))
        inv = int(gf_inv(np.uint8(v[pivot_col])))
        v = gf_mul(inv, v)
        p = gf_mul(inv, p)

        self._coeff_rows.append(v)
        self._payload_rows.append(p)
        return True

    def decode(self) -> np.ndarray | None:
        """Return the recovered (K, payload_bytes) source-packet matrix if the
        generation is fully decodable, else None."""
        if not self.is_decoded:
            return None
        A = np.stack(self._coeff_rows)[: self.K].astype(np.int64)
        B = np.stack(self._payload_rows)[: self.K].astype(np.uint8)
        # Back-substitute to full reduced row echelon form so each row's
        # pivot column is unique and equals its row index (rows arrive with
        # pivots discovered in receive-order, not sorted order).
        order = np.argmax(A != 0, axis=1)
        perm = np.argsort(order)
        A, B = A[perm], B[perm]
        for i in range(self.K):
            for j in range(self.K):
                if i != j and A[j, i] != 0:
                    factor = A[j, i]
                    A[j, :] = gf_add(A[j, :], gf_mul(factor, A[i, :]))
                    B[j, :] = gf_add(B[j, :], gf_mul(factor, B[i, :]))
        return B


def decode_probability(K: int, R: int, p_erasure: float | np.ndarray) -> np.ndarray:
    """
    Closed-form probability that a receiver decodes a generation of K source
    packets given K + R packets are transmitted, each independently erased
    with probability `p_erasure` (APC-RLNC Eq. (3)):

        P_decode = sum_{k=K}^{K+R} C(K+R, k) (1-p)^k p^(K+R-k)

    This treats "receiving >= K of the K+R transmitted packets" as a
    necessary condition for decodability; because coding coefficients are
    drawn uniformly from GF(256)\\{0} (256 >> K in all configurations used
    here), the probability that K arbitrary coded packets are linearly
    dependent is smaller than 1/255 and is neglected, consistent with the
    standard RLNC decoding-probability approximation (Ho et al., 2006).

    Vectorized over `p_erasure` (scalar or ndarray).
    """
    p_in = np.asarray(p_erasure, dtype=np.float64)
    p_flat = np.clip(p_in.reshape(-1), 0.0, 1.0)
    n = K + R
    ks = np.arange(K, n + 1)
    pmf = stats.binom.pmf(ks[:, None], n, (1.0 - p_flat)[None, :])  # (n-K+1, M)
    result = pmf.sum(axis=0)
    if p_in.ndim == 0:
        return float(result[0])
    return result.reshape(p_in.shape)


def decode_probability_mc(
    K: int, R: int, p_erasure: float, n_trials: int = 20_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte-Carlo estimate of P_decode, used only in tests to cross-check
    `decode_probability` against an independent, simulation-based estimate
    (including the small linear-dependence correction that the closed form
    neglects)."""
    rng = rng or np.random.default_rng()
    n = K + R
    successes = 0
    for _ in range(n_trials):
        erased = rng.random(n) < p_erasure
        n_received = int(np.sum(~erased))
        if n_received < K:
            continue
        # Approximate: if >= K packets are received, decoding succeeds
        # unless the received coding vectors happen to be rank-deficient,
        # which for GF(256) with K <= 64 has probability < 1/255 and is
        # sampled explicitly here for a faithful MC comparison.
        coeffs = rng.integers(1, 256, size=(n_received, K)).astype(np.uint8)
        if GF256.rank(coeffs) >= K:
            successes += 1
    return successes / n_trials
