"""
Unified state-space engine for PIR-Framework.

Formalizes the joint state vector

    X_t = [S_m(t), S_w(t), S_e(t), S_c(t)]

where:

  S_m  Medical      -- context-aware combination of per-modality Signal
                        Quality Index (SQI) and a derived scalar
                        Physiological Information Value (PIV in [0,1]).
  S_w  Wireless      -- instantaneous SNR (dB) and burst-erasure probability
                        of a two-state Gilbert-Elliott Markov channel.
  S_e  Energy        -- residual battery capacity (J) and joules-per-packet
                        expenditure, following an APC-RLNC-style additive
                        energy model (E = E_tx*n_tx + E_rx*n_rx +
                        E_comp*n_ops), re-parameterized to body-worn-sensor
                        scale so a <100 mW average-power constraint is
                        physically meaningful.
  S_c  Compute       -- clock-cycle latency model for on-device ("local")
                        vs. offloaded ("edge") inference, plus the
                        transmission/propagation latency implied by the
                        chosen RLNC redundancy and channel state.

Everything here is vectorized over a batch of N independent
environments/windows so that the contextual bandit in `pir_controller.py`
can be trained and evaluated over thousands of windows without a Python-level
per-window loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .channel import GilbertElliottChannel

# --------------------------------------------------------------------------
# S_m : Medical state
# --------------------------------------------------------------------------


@dataclass
class MedicalState:
    """Per-window medical state for a batch of N windows.

    sqi_ecg, sqi_ppg : Orphanidou-type signal-quality index in [0, 1]; use
                        0.0 (not NaN) to represent a fully-missing modality
                        so downstream arithmetic stays vectorized.
    modality_present  : boolean masks, True if the modality's sensor is
                        physically attached/streaming this window (distinct
                        from *quality* -- a present-but-degraded modality has
                        modality_present=True and a low sqi).
    hr_ecg, hr_ppg    : per-modality heart-rate estimates (bpm); used only to
                        compute cross-modal agreement, not as ground truth.
    """

    sqi_ecg: np.ndarray
    sqi_ppg: np.ndarray
    modality_present_ecg: np.ndarray
    modality_present_ppg: np.ndarray
    hr_ecg: np.ndarray
    hr_ppg: np.ndarray
    piv: np.ndarray = field(init=False)

    HR_RANGE_BPM: float = 140.0  # U(50, 110) generator range width used elsewhere -> normalizer
    W_QUALITY: float = 0.75
    W_AGREEMENT: float = 0.25

    def __post_init__(self) -> None:
        self.piv = compute_piv(
            self.sqi_ecg, self.sqi_ppg,
            self.modality_present_ecg, self.modality_present_ppg,
            self.hr_ecg, self.hr_ppg,
            hr_range_bpm=self.HR_RANGE_BPM,
            w_quality=self.W_QUALITY, w_agreement=self.W_AGREEMENT,
        )

    def as_array(self) -> np.ndarray:
        """Feature columns: [sqi_ecg, sqi_ppg, present_ecg, present_ppg, piv]."""
        return np.stack(
            [self.sqi_ecg, self.sqi_ppg,
             self.modality_present_ecg.astype(np.float64),
             self.modality_present_ppg.astype(np.float64),
             self.piv],
            axis=-1,
        )


def compute_piv(
    sqi_ecg: np.ndarray, sqi_ppg: np.ndarray,
    present_ecg: np.ndarray, present_ppg: np.ndarray,
    hr_ecg: np.ndarray, hr_ppg: np.ndarray,
    hr_range_bpm: float = 140.0, w_quality: float = 0.75, w_agreement: float = 0.25,
) -> np.ndarray:
    """
    Physiological Information Value (PIV) in [0, 1]: how much clinically
    trustworthy information this window's fused ECG/PPG estimate carries.

    PIV = w_quality * SQI_combined - w_agreement * disagreement_penalty,
    clipped to [0, 1], where:

      * SQI_combined is a *presence-aware* weighted average of the two
        modalities' SQI (a missing modality contributes 0 quality and 0
        weight, so it is excluded rather than dragging the average down
        with a spurious 0);
      * disagreement_penalty is the normalized cross-modal heart-rate
        disagreement when *both* modalities are present (0 if either
        modality is missing, since there is nothing to disagree with).

    This mirrors CardioFusion-AI's finding that modality *availability* and
    modality *quality* are functionally distinct signals: PIV therefore
    combines a quality term (from SQI) with an independent, availability-
    gated agreement term, rather than conflating the two into a single
    heuristic score.
    """
    sqi_ecg = np.asarray(sqi_ecg, dtype=np.float64)
    sqi_ppg = np.asarray(sqi_ppg, dtype=np.float64)
    present_ecg = np.asarray(present_ecg, dtype=bool)
    present_ppg = np.asarray(present_ppg, dtype=bool)

    w_ecg = np.where(present_ecg, sqi_ecg, 0.0)
    w_ppg = np.where(present_ppg, sqi_ppg, 0.0)
    weight_sum = w_ecg + w_ppg
    # If both weights are ~0 (both modalities missing or both SQI==0),
    # SQI_combined is 0 by definition (no informative modality at all).
    safe_denom = np.where(weight_sum > 1e-9, weight_sum, 1.0)
    sqi_combined = np.where(
        weight_sum > 1e-9,
        (w_ecg * np.where(present_ecg, sqi_ecg, 0.0) + w_ppg * np.where(present_ppg, sqi_ppg, 0.0)) / safe_denom,
        0.0,
    )

    both_present = present_ecg & present_ppg
    disagreement = np.abs(hr_ecg - hr_ppg) / hr_range_bpm
    disagreement_penalty = np.where(both_present, disagreement, 0.0)

    piv = w_quality * sqi_combined - w_agreement * disagreement_penalty
    return np.clip(piv, 0.0, 1.0)


# --------------------------------------------------------------------------
# S_w : Wireless state -- thin re-export of the channel step, kept here so
# the joint state vector is assembled from one place.
# --------------------------------------------------------------------------


class WirelessState(NamedTuple):
    link_state: np.ndarray   # 0 = GOOD, 1 = BAD
    snr_db: np.ndarray
    ber: np.ndarray
    p_erasure: np.ndarray

    def as_array(self) -> np.ndarray:
        return np.stack([self.snr_db, self.p_erasure, self.link_state.astype(np.float64)], axis=-1)


# --------------------------------------------------------------------------
# S_e : Energy state
# --------------------------------------------------------------------------


@dataclass
class EnergyModel:
    """Body-worn-sensor-scale additive energy model, structurally identical
    to the APC-RLNC energy accounting (E = E_tx n_tx + E_rx n_rx +
    E_comp * n_ops) but re-parameterized to joules-per-packet magnitudes
    appropriate for a BLE-class radio and a low-power MCU, so that a
    100 mW average-power cap is a real constraint rather than either
    vacuous or unsatisfiable.
    """

    e_tx_j: float = 2.0e-3     # energy per transmitted coded packet (BLE-class radio)
    e_rx_j: float = 1.0e-3     # energy per received ACK/feedback packet
    e_comp_per_op_j: float = 2.0e-7  # energy per GF(256) MAC-equivalent operation
    e_idle_w: float = 1.0e-3   # baseline sensing + MCU idle draw

    def compute(
        self, n_tx: np.ndarray, n_rx: np.ndarray, n_ops: np.ndarray, window_seconds: float,
    ) -> np.ndarray:
        """Total energy (J) consumed this window."""
        return (
            self.e_tx_j * n_tx
            + self.e_rx_j * n_rx
            + self.e_comp_per_op_j * n_ops
            + self.e_idle_w * window_seconds
        )


class EnergyState:
    """Per-environment residual battery, tracked across steps."""

    def __init__(self, n_envs: int, capacity_j: float) -> None:
        self.capacity_j = capacity_j
        self.battery_j = np.full(n_envs, capacity_j, dtype=np.float64)

    @staticmethod
    def initialize(n_envs: int, capacity_j: float) -> "EnergyState":
        return EnergyState(n_envs, capacity_j)

    def consume(self, energy_j: np.ndarray) -> None:
        self.battery_j = np.maximum(self.battery_j - energy_j, 0.0)

    def fraction_remaining(self) -> np.ndarray:
        return self.battery_j / self.capacity_j

    def as_array(self, energy_j: np.ndarray, power_mw: np.ndarray) -> np.ndarray:
        return np.stack([self.fraction_remaining(), energy_j, power_mw], axis=-1)


# --------------------------------------------------------------------------
# S_c : Compute / latency state
# --------------------------------------------------------------------------


@dataclass
class ComputeModel:
    """Clock-cycle latency model for local (on-sensor) vs. edge-offloaded
    inference, plus RLNC transmission latency.

    local_clock_hz    : on-body MCU clock (e.g. 64 MHz Cortex-M class).
    edge_clock_hz      : offload target clock (e.g. 1.4 GHz Cortex-A class,
                          matching the APC-RLNC Jetson Nano testbed).
    cycles_per_gf_op   : clock cycles per GF(256) multiply-accumulate.
    cycles_per_local_infer : fixed cost of the on-device physiological
                          estimator (peak detection + SQI + regression head).
    phy_rate_bps       : effective PHY data rate for packet transmission.
    rtt_ms             : fixed round-trip propagation latency to the edge
                          node, only paid when placement == "edge".
    """

    local_clock_hz: float = 64e6
    edge_clock_hz: float = 1.4e9
    cycles_per_gf_op: float = 8.0
    cycles_per_local_infer: float = 2.0e6
    cycles_per_edge_infer: float = 5.0e6
    phy_rate_bps: float = 250e3  # BLE 5 long-range-ish effective throughput
    rtt_ms: float = 12.0

    def encode_decode_ops(self, K: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Approximate GF(256) op count: encoding is O(K*(K+R)) MACs per
        payload byte-column bundle (amortized as a scalar op count here);
        decoding (Gaussian elimination) is O(K^3)."""
        K = np.asarray(K, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        encode_ops = K * (K + R)
        decode_ops = K ** 3
        return encode_ops + decode_ops

    def transmission_latency_ms(self, K: np.ndarray, R: np.ndarray, payload_bits: float) -> np.ndarray:
        n_coded = np.asarray(K, dtype=np.float64) + np.asarray(R, dtype=np.float64)
        total_bits = n_coded * payload_bits
        return (total_bits / self.phy_rate_bps) * 1000.0

    def compute_latency_ms(
        self, K: np.ndarray, R: np.ndarray, placement_is_edge: np.ndarray, payload_bits: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (total_latency_ms, compute_only_latency_ms)."""
        ops = self.encode_decode_ops(K, R)
        clock = np.where(placement_is_edge, self.edge_clock_hz, self.local_clock_hz)
        base_infer_cycles = np.where(
            placement_is_edge, self.cycles_per_edge_infer, self.cycles_per_local_infer,
        )
        compute_cycles = ops * self.cycles_per_gf_op + base_infer_cycles
        compute_ms = (compute_cycles / clock) * 1000.0

        tx_ms = self.transmission_latency_ms(K, R, payload_bits)
        rtt_ms = np.where(placement_is_edge, self.rtt_ms, 0.0)
        total_ms = compute_ms + tx_ms + rtt_ms
        return total_ms, compute_ms


# --------------------------------------------------------------------------
# Joint step result
# --------------------------------------------------------------------------


class StepResult(NamedTuple):
    medical: MedicalState
    wireless: WirelessState
    energy_j: np.ndarray
    power_mw: np.ndarray
    battery_fraction: np.ndarray
    latency_ms: np.ndarray
    compute_ms: np.ndarray
    p_decode: np.ndarray
    context: np.ndarray  # (N, D) feature vector for the bandit


class StateSpaceEngine:
    """
    Drives the joint state X_t = [S_m, S_w, S_e, S_c] forward one epoch at a
    time for a batch of N parallel environments (windows).

    The medical state S_m is supplied externally each step (it comes from
    real/synthetic physiological windows produced by `data_pipeline.py`);
    the wireless state S_w evolves autonomously via a Gilbert-Elliott
    channel; the energy and compute states S_e, S_c are deterministic
    functions of the *action* chosen by the controller (K, R, placement)
    and the current wireless state.
    """

    def __init__(
        self,
        n_envs: int,
        battery_capacity_j: float = 20.0,
        window_seconds: float = 8.0,
        payload_bits: float = 8 * 64,  # 64-byte coded payload (compact physiological feature packet)
        channel_kwargs: dict | None = None,
        energy_model: EnergyModel | None = None,
        compute_model: ComputeModel | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_envs = n_envs
        self.window_seconds = window_seconds
        self.payload_bits = payload_bits
        self.channel = GilbertElliottChannel(n_links=n_envs, seed=seed, **(channel_kwargs or {}))
        self.energy_model = energy_model or EnergyModel()
        self.compute_model = compute_model or ComputeModel()
        self.energy_state = EnergyState.initialize(n_envs, battery_capacity_j)
        init_raw = self.channel.step()
        self.last_wireless = WirelessState(
            link_state=init_raw["state"], snr_db=init_raw["snr_db"],
            ber=init_raw["ber"], p_erasure=init_raw["p_erasure"],
        )
        self._prev_p_erasure_ewma = np.full(n_envs, self.channel.stationary_bad_prob())
        self.ewma_alpha = 0.3

    def observe_context(self, medical: MedicalState) -> np.ndarray:
        """Return the context vector the controller decides on, built from
        the *previously observed* wireless telemetry (`self.last_wireless`)
        -- i.e. the channel state as of the last feedback report -- rather
        than the channel realization the about-to-be-chosen action will
        actually be transmitted over. This mirrors real adaptive coding
        systems, which set redundancy from the most recent channel
        estimate, not from a not-yet-realized future draw, and prevents an
        unfair "oracle knows this round's exact erasure outcome" leak into
        the bandit's decision inputs."""
        return self.build_context(medical, self.last_wireless)

    def act(
        self,
        medical: MedicalState,
        K: np.ndarray,
        R: np.ndarray,
        placement_is_edge: np.ndarray,
        n_feedback_packets: np.ndarray | None = None,
    ) -> StepResult:
        """Advance the wireless channel by one epoch (drawing this round's
        *actual* realization, only knowable after the transmission is
        attempted) and compute the energy/latency/decode outcome implied by
        the chosen action against that realization. Updates
        `self.last_wireless` for the *next* call to `observe_context`."""
        raw = self.channel.step()
        wireless = WirelessState(
            link_state=raw["state"], snr_db=raw["snr_db"], ber=raw["ber"], p_erasure=raw["p_erasure"],
        )
        self._prev_p_erasure_ewma = (
            self.ewma_alpha * wireless.p_erasure + (1 - self.ewma_alpha) * self._prev_p_erasure_ewma
        )

        n_tx = K + R
        n_rx = np.asarray(n_feedback_packets if n_feedback_packets is not None else np.ones_like(K))
        n_ops = self.compute_model.encode_decode_ops(K, R)
        energy_j = self.energy_model.compute(n_tx, n_rx, n_ops, self.window_seconds)
        self.energy_state.consume(energy_j)
        power_mw = (energy_j / self.window_seconds) * 1000.0

        latency_ms, compute_ms = self.compute_model.compute_latency_ms(
            K, R, placement_is_edge, self.payload_bits,
        )

        p_decode = _batched_decode_probability(K, R, wireless.p_erasure)

        context = self.build_context(medical, wireless)
        self.last_wireless = wireless
        return StepResult(
            medical=medical, wireless=wireless, energy_j=energy_j, power_mw=power_mw,
            battery_fraction=self.energy_state.fraction_remaining(),
            latency_ms=latency_ms, compute_ms=compute_ms, p_decode=p_decode, context=context,
        )

    def build_context(self, medical: MedicalState, wireless: WirelessState) -> np.ndarray:
        """Assemble the bandit's context vector from the joint state.

        Columns: [sqi_ecg, sqi_ppg, present_ecg, present_ppg, piv,
                  snr_db (normalized), p_erasure_ewma, link_state,
                  battery_fraction, bias=1.0]
        """
        m = medical.as_array()  # (N, 5)
        snr_norm = np.clip(wireless.snr_db / 20.0, -1.0, 2.0)
        battery_frac = self.energy_state.fraction_remaining()
        bias = np.ones(self.n_envs)
        return np.concatenate(
            [m, snr_norm[:, None], self._prev_p_erasure_ewma[:, None],
             wireless.link_state.astype(np.float64)[:, None], battery_frac[:, None], bias[:, None]],
            axis=-1,
        )

    @staticmethod
    def context_feature_names() -> list[str]:
        return [
            "sqi_ecg", "sqi_ppg", "present_ecg", "present_ppg", "piv",
            "snr_db_norm", "p_erasure_ewma", "link_state", "battery_fraction", "bias",
        ]


def _batched_decode_probability(K: np.ndarray, R: np.ndarray, p_erasure: np.ndarray) -> np.ndarray:
    """Per-environment decode probability where K, R may vary across the
    batch (unlike `coding.rlnc.decode_probability`, which assumes a single
    (K, R) pair). Groups by unique (K, R) pairs for efficiency."""
    from .coding.rlnc import decode_probability

    K = np.asarray(K)
    R = np.asarray(R)
    p_erasure = np.asarray(p_erasure, dtype=np.float64)
    out = np.empty_like(p_erasure)
    pairs = np.stack([K, R], axis=-1)
    unique_pairs = np.unique(pairs, axis=0)
    for k_val, r_val in unique_pairs:
        mask = (K == k_val) & (R == r_val)
        out[mask] = decode_probability(int(k_val), int(r_val), p_erasure[mask])
    return out
