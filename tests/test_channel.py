import numpy as np
import pytest

from pir_framework.channel.gilbert_elliott import (
    GilbertElliottChannel, snr_to_ber_bpsk, ber_to_packet_erasure, GOOD, BAD,
)


class TestSNRModel:
    def test_ber_decreases_with_snr(self):
        snr = np.array([-5, 0, 5, 10, 20])
        ber = snr_to_ber_bpsk(snr)
        assert np.all(np.diff(ber) < 0)  # strictly decreasing

    def test_ber_bounds(self):
        ber = snr_to_ber_bpsk(np.array([-50, 50]))
        assert 0.0 <= ber[0] <= 1.0
        assert 0.0 <= ber[1] <= 1.0
        assert ber[1] < 1e-10  # very high SNR -> near-zero BER

    def test_packet_erasure_increases_with_payload_length(self):
        ber = 0.01
        small = ber_to_packet_erasure(ber, payload_bits=8)
        large = ber_to_packet_erasure(ber, payload_bits=800)
        assert large > small

    def test_packet_erasure_zero_ber(self):
        assert ber_to_packet_erasure(0.0, payload_bits=1000) == pytest.approx(0.0)


class TestGilbertElliottChannel:
    def test_stationary_distribution_matches_theory(self):
        ch = GilbertElliottChannel(n_links=50000, p_gb=0.05, p_bg=0.15, seed=0)
        out = ch.simulate(50)
        # After many steps, fraction of time in BAD should approach
        # p_gb / (p_gb + p_bg) = 0.05/0.20 = 0.25
        empirical_bad_frac = np.mean(out["state"][-1] == BAD)
        assert abs(empirical_bad_frac - 0.25) < 0.03

    def test_good_state_has_lower_erasure_than_bad_state(self):
        ch = GilbertElliottChannel(n_links=20000, seed=1)
        out = ch.simulate(10)
        good_mask = out["state"] == GOOD
        bad_mask = out["state"] == BAD
        mean_good = out["p_erasure"][good_mask].mean()
        mean_bad = out["p_erasure"][bad_mask].mean()
        assert mean_good < mean_bad

    def test_erasure_probability_in_valid_range(self):
        ch = GilbertElliottChannel(n_links=1000, seed=2)
        out = ch.simulate(20)
        assert np.all(out["p_erasure"] >= 0.0)
        assert np.all(out["p_erasure"] <= 1.0)

    def test_reproducible_with_same_seed(self):
        ch1 = GilbertElliottChannel(n_links=100, seed=42)
        ch2 = GilbertElliottChannel(n_links=100, seed=42)
        out1 = ch1.simulate(30)
        out2 = ch2.simulate(30)
        assert np.array_equal(out1["snr_db"], out2["snr_db"])
        assert np.array_equal(out1["state"], out2["state"])

    def test_different_seeds_diverge(self):
        ch1 = GilbertElliottChannel(n_links=100, seed=1)
        ch2 = GilbertElliottChannel(n_links=100, seed=2)
        out1 = ch1.simulate(30)
        out2 = ch2.simulate(30)
        assert not np.array_equal(out1["snr_db"], out2["snr_db"])

    def test_step_shapes(self):
        n = 37
        ch = GilbertElliottChannel(n_links=n, seed=3)
        out = ch.step()
        for key in ("state", "snr_db", "ber", "p_erasure"):
            assert out[key].shape == (n,)
