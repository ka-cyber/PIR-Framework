"""
Evaluation metrics used by `evaluate.py` to compare policies:
Heart-Rate MAE, clinical-event-detection F1, Packet Delivery Ratio (PDR),
per-window energy consumption, deadline-success rate, and mean reward.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


@dataclass
class EvaluationMetrics:
    policy: str
    regime: str
    n_windows: int
    hr_mae_bpm: float
    event_f1: float
    pdr: float
    mean_energy_j: float
    mean_power_mw: float
    mean_latency_ms: float
    deadline_success_rate: float
    power_cap_success_rate: float
    mean_reward: float
    mean_piv: float
    mean_u_pir: float

    def to_dict(self) -> dict:
        return asdict(self)


def event_labels_from_hr(hr_bpm: np.ndarray, brady_thresh: float = 60.0, tachy_thresh: float = 100.0) -> np.ndarray:
    """Binary clinical-event flag: 1 if the (true or delivered) heart rate
    indicates bradycardia or tachycardia, 0 for a normal-range window. Used
    as a simple, transparent downstream clinical task on top of the raw HR
    estimate, so degraded/undelivered windows have a measurable clinical
    consequence (a missed or false alarm) rather than only a numeric MAE."""
    hr_bpm = np.asarray(hr_bpm, dtype=np.float64)
    return ((hr_bpm <= brady_thresh) | (hr_bpm >= tachy_thresh)).astype(int)


def compute_window_metrics(
    policy: str,
    regime: str,
    hr_true_bpm: np.ndarray,
    hr_delivered_bpm: np.ndarray,
    decode_success: np.ndarray,
    energy_j: np.ndarray,
    power_mw: np.ndarray,
    latency_ms: np.ndarray,
    reward: np.ndarray,
    piv: np.ndarray,
    u_pir: np.ndarray,
    deadline_ms: float = 150.0,
    power_cap_mw: float = 100.0,
) -> EvaluationMetrics:
    """
    Aggregate a batch of per-window outcomes for one (policy, regime) cell
    into a single `EvaluationMetrics` record.

    `hr_delivered_bpm` should already encode the policy's fallback behavior
    for undelivered windows (e.g. holding the last successfully-delivered
    estimate); this function only aggregates, it does not implement the
    fallback logic itself (see `evaluate.py`).
    """
    hr_true_bpm = np.asarray(hr_true_bpm, dtype=np.float64)
    hr_delivered_bpm = np.asarray(hr_delivered_bpm, dtype=np.float64)
    decode_success = np.asarray(decode_success, dtype=bool)

    hr_mae = float(np.mean(np.abs(hr_true_bpm - hr_delivered_bpm)))

    y_true_events = event_labels_from_hr(hr_true_bpm)
    y_pred_events = event_labels_from_hr(hr_delivered_bpm)
    if y_true_events.sum() == 0 and y_pred_events.sum() == 0:
        f1 = 1.0  # no events, none predicted: trivially perfect agreement
    elif y_true_events.sum() == 0 or y_pred_events.sum() == 0:
        f1 = float(f1_score(y_true_events, y_pred_events, zero_division=0))
    else:
        f1 = float(f1_score(y_true_events, y_pred_events))

    pdr = float(np.mean(decode_success))
    mean_energy = float(np.mean(energy_j))
    mean_power = float(np.mean(power_mw))
    mean_latency = float(np.mean(latency_ms))
    deadline_rate = float(np.mean(latency_ms <= deadline_ms))
    power_rate = float(np.mean(power_mw <= power_cap_mw))
    mean_reward = float(np.mean(reward))
    mean_piv = float(np.mean(piv))
    mean_u_pir = float(np.mean(u_pir))

    return EvaluationMetrics(
        policy=policy, regime=regime, n_windows=int(len(hr_true_bpm)),
        hr_mae_bpm=hr_mae, event_f1=f1, pdr=pdr,
        mean_energy_j=mean_energy, mean_power_mw=mean_power, mean_latency_ms=mean_latency,
        deadline_success_rate=deadline_rate, power_cap_success_rate=power_rate,
        mean_reward=mean_reward, mean_piv=mean_piv, mean_u_pir=mean_u_pir,
    )


def aggregate_metrics(records: list[EvaluationMetrics]) -> pd.DataFrame:
    """Convert a list of `EvaluationMetrics` into a tidy DataFrame."""
    return pd.DataFrame([r.to_dict() for r in records])


def overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-(policy, regime) metrics table to per-policy overall
    metrics, weighting each regime's contribution by its window count."""
    rows = []
    for policy, g in df.groupby("policy"):
        w = g["n_windows"].to_numpy(dtype=np.float64)
        w_sum = w.sum()
        row = {"policy": policy, "regime": "overall", "n_windows": int(w_sum)}
        for col in [
            "hr_mae_bpm", "event_f1", "pdr", "mean_energy_j", "mean_power_mw",
            "mean_latency_ms", "deadline_success_rate", "power_cap_success_rate",
            "mean_reward", "mean_piv", "mean_u_pir",
        ]:
            row[col] = float(np.average(g[col].to_numpy(dtype=np.float64), weights=w))
        rows.append(row)
    return pd.DataFrame(rows)
