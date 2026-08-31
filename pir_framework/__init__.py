"""
PIR-Framework: Physiological Information Reliability
======================================================

A unified state-space and contextual-bandit control framework that bridges
two prior systems:

  * CardioFusion-AI  -- multimodal ECG/PPG signal-quality (SQI) and
                          physiological-information-value (PIV) estimation.
  * APC-RLNC          -- adaptive Random Linear Network Coding (RLNC) over
                          GF(2^8) under bursty Gilbert-Elliott erasure
                          channels, with energy and compute-latency models.

PIR-Framework treats "how much clinically-relevant information survives
the end-to-end sense -> encode -> transmit -> decode -> act pipeline,
per joule and per millisecond of latency budget" as the object to be
actively controlled, using a contextual bandit rather than fixed or
hand-tuned heuristic policies.

Sub-packages
------------
state_space      Joint state vector X_t = [S_m, S_w, S_e, S_c] and a
                 vectorized (NumPy) multi-environment simulator.
pir_controller   LinUCB / Thompson-Sampling contextual bandit optimizing a
                 constrained multi-objective reward.
data_pipeline    PhysioNet-format-compatible data loader + physiologically
                 grounded synthetic ECG/PPG generator and artifact injector.
evaluate         Comparative evaluation harness against baseline policies,
                 with LaTeX table and Matplotlib figure export.
coding           GF(256) arithmetic and RLNC encode/decode probability.
channel          Gilbert-Elliott two-state Markov erasure channel with an
                 SNR-linked BER model.
utils            Metrics, LaTeX export, and plotting helpers.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("pir-framework")
except PackageNotFoundError:  # pragma: no cover - local/dev installs
    __version__ = "0.1.0"

__all__ = ["__version__"]
