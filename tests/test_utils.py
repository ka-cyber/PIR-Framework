import numpy as np
import pandas as pd
import pytest

from pir_framework.utils.metrics import (
    compute_window_metrics, aggregate_metrics, overall_metrics, event_labels_from_hr,
)
from pir_framework.utils.latex_export import dataframe_to_latex_table


class TestEventLabels:
    def test_bradycardia_and_tachycardia_flagged(self):
        hr = np.array([45.0, 75.0, 110.0])
        labels = event_labels_from_hr(hr, brady_thresh=60, tachy_thresh=100)
        assert list(labels) == [1, 0, 1]

    def test_normal_range_not_flagged(self):
        hr = np.array([60.0, 100.0])  # boundary values are flagged (<=, >=)
        labels = event_labels_from_hr(hr, brady_thresh=60, tachy_thresh=100)
        assert list(labels) == [1, 1]


class TestComputeWindowMetrics:
    def _make_inputs(self, n=50, seed=0):
        rng = np.random.default_rng(seed)
        hr_true = rng.uniform(50, 110, n)
        hr_delivered = hr_true + rng.normal(0, 2, n)
        decode_success = rng.random(n) > 0.2
        energy = rng.uniform(0.05, 0.15, n)
        power = rng.uniform(5, 15, n)
        latency = rng.uniform(50, 140, n)
        reward = rng.normal(0.2, 0.1, n)
        piv = rng.uniform(0, 1, n)
        u_pir = piv * decode_success
        return hr_true, hr_delivered, decode_success, energy, power, latency, reward, piv, u_pir

    def test_perfect_delivery_gives_pdr_one(self):
        n = 20
        hr_true = np.full(n, 70.0)
        decode_success = np.ones(n, dtype=bool)
        m = compute_window_metrics(
            "P", "R", hr_true, hr_true, decode_success,
            np.full(n, 0.1), np.full(n, 10.0), np.full(n, 100.0),
            np.full(n, 0.2), np.full(n, 0.7), np.full(n, 0.7),
        )
        assert m.pdr == pytest.approx(1.0)
        assert m.hr_mae_bpm == pytest.approx(0.0)

    def test_deadline_and_power_cap_success_rates(self):
        n = 10
        latency = np.array([100.0] * 5 + [200.0] * 5)  # 5 under, 5 over 150ms deadline
        power = np.array([50.0] * 3 + [150.0] * 7)      # 3 under, 7 over 100mW cap
        m = compute_window_metrics(
            "P", "R", np.full(n, 70.0), np.full(n, 70.0), np.ones(n, dtype=bool),
            np.full(n, 0.1), power, latency, np.full(n, 0.2), np.full(n, 0.7), np.full(n, 0.7),
            deadline_ms=150.0, power_cap_mw=100.0,
        )
        assert m.deadline_success_rate == pytest.approx(0.5)
        assert m.power_cap_success_rate == pytest.approx(0.3)

    def test_aggregate_and_overall_metrics_roundtrip(self):
        args = self._make_inputs()
        m1 = compute_window_metrics("PolicyA", "regime1", *args)
        m2 = compute_window_metrics("PolicyA", "regime2", *args)
        df = aggregate_metrics([m1, m2])
        assert len(df) == 2
        overall = overall_metrics(df)
        assert len(overall) == 1
        assert overall.iloc[0]["policy"] == "PolicyA"
        assert overall.iloc[0]["n_windows"] == m1.n_windows + m2.n_windows


class TestLatexExport:
    def test_basic_table_structure(self):
        df = pd.DataFrame({
            "policy": ["A", "B"], "hr_mae_bpm": [5.0, 6.0], "pdr": [0.9, 0.8],
        })
        tex = dataframe_to_latex_table(df, columns=["hr_mae_bpm", "pdr"])
        assert "\\begin{table}" in tex
        assert "\\end{table}" in tex
        assert "\\toprule" in tex and "\\bottomrule" in tex
        assert "A &" in tex and "B &" in tex

    def test_bold_best_marks_correct_row(self):
        df = pd.DataFrame({
            "policy": ["A", "B"], "hr_mae_bpm": [5.0, 3.0],
        })
        tex = dataframe_to_latex_table(
            df, columns=["hr_mae_bpm"], bold_best={"hr_mae_bpm": "min"},
        )
        lines = tex.split("\n")
        b_line = next(l for l in lines if l.startswith("B "))
        a_line = next(l for l in lines if l.startswith("A "))
        assert "\\textbf" in b_line  # B has the lower (better) MAE
        assert "\\textbf" not in a_line

    def test_ci_formatting(self):
        df = pd.DataFrame({"policy": ["A"], "hr_mae_bpm": [5.0]})
        ci = pd.DataFrame({"policy": ["A"], "hr_mae_bpm": [0.5]})
        tex = dataframe_to_latex_table(df, columns=["hr_mae_bpm"], ci_df=ci)
        assert "$\\pm$" in tex

    def test_policy_name_underscore_escaped(self):
        df = pd.DataFrame({"policy": ["Fixed_Rate"], "hr_mae_bpm": [5.0]})
        tex = dataframe_to_latex_table(df, columns=["hr_mae_bpm"])
        assert "Fixed\\_Rate" in tex
