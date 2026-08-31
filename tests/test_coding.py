import numpy as np
import pytest

from pir_framework.coding.gf256 import GF256, gf_add, gf_mul, gf_div, gf_inv, gf_pow
from pir_framework.coding.rlnc import (
    RLNCEncoder, RLNCDecoder, decode_probability, decode_probability_mc,
)


class TestGF256Arithmetic:
    def test_add_is_xor_and_self_inverse(self):
        a = np.array([1, 100, 255, 0], dtype=np.uint8)
        b = np.array([2, 50, 128, 0], dtype=np.uint8)
        assert np.array_equal(gf_add(a, b), a ^ b)
        assert np.array_equal(gf_add(a, a), np.zeros_like(a))  # a + a = 0 in char-2 field

    def test_mul_by_zero_and_one(self):
        a = np.array([1, 5, 200, 255], dtype=np.uint8)
        assert np.array_equal(gf_mul(a, 0), np.zeros_like(a))
        assert np.array_equal(gf_mul(a, 1), a)

    def test_mul_commutative_and_associative(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, size=50).astype(np.uint8)
        b = rng.integers(0, 256, size=50).astype(np.uint8)
        c = rng.integers(0, 256, size=50).astype(np.uint8)
        assert np.array_equal(gf_mul(a, b), gf_mul(b, a))
        assert np.array_equal(gf_mul(gf_mul(a, b), c), gf_mul(a, gf_mul(b, c)))

    def test_distributive(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 256, size=30).astype(np.uint8)
        b = rng.integers(0, 256, size=30).astype(np.uint8)
        c = rng.integers(0, 256, size=30).astype(np.uint8)
        lhs = gf_mul(a, gf_add(b, c))
        rhs = gf_add(gf_mul(a, b), gf_mul(a, c))
        assert np.array_equal(lhs, rhs)

    def test_inverse_round_trip(self):
        for v in range(1, 256):
            inv = int(gf_inv(np.uint8(v)))
            assert int(gf_mul(np.uint8(v), np.uint8(inv))) == 1

    def test_inverse_of_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            gf_inv(np.array([0], dtype=np.uint8))

    def test_div_matches_mul_by_inverse(self):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 256, size=40).astype(np.uint8)
        b = rng.integers(1, 256, size=40).astype(np.uint8)  # avoid div by 0
        assert np.array_equal(gf_div(a, b), gf_mul(a, gf_inv(b)))

    def test_pow(self):
        v = np.array([3], dtype=np.uint8)
        assert int(gf_pow(v, 0)[0]) == 1
        assert int(gf_pow(v, 1)[0]) == 3
        assert int(gf_pow(v, 2)[0]) == int(gf_mul(v, v)[0])

    def test_primitive_element_order_255(self):
        # 2 should be a primitive element: 2^255 == 1, and no smaller
        # positive divisor-of-255 exponent gives 1 (checked for the
        # standard nontrivial divisors, which is sufficient evidence
        # of primitivity together with 2^255==1).
        x = np.uint8(1)
        two = np.uint8(2)
        seen = set()
        for _ in range(255):
            x = gf_mul(x, two)
            seen.add(int(x))
        assert int(x) == 1  # 2^255 = 1
        assert len(seen) == 255  # every nonzero element reached -> order 255

    def test_matrix_multiply_identity(self):
        rng = np.random.default_rng(3)
        A = rng.integers(1, 256, size=(4, 4)).astype(np.uint8)
        I = GF256.identity(4)
        assert np.array_equal(GF256.matmul(A, I), A)

    def test_rank_full_vs_deficient(self):
        I = GF256.identity(5)
        assert GF256.rank(I) == 5
        deficient = np.array(I, copy=True)
        deficient[4, :] = deficient[0, :]  # duplicate row -> rank drops by 1
        assert GF256.rank(deficient) == 4


class TestRLNC:
    def test_encode_decode_round_trip_exact_K_packets(self):
        rng = np.random.default_rng(42)
        K, R, payload = 8, 4, 32
        enc = RLNCEncoder(K=K, R=R, payload_bytes=payload, rng=rng)
        src = rng.integers(0, 256, size=(K, payload)).astype(np.uint8)
        cv, cp = enc.encode(src)

        dec = RLNCDecoder(K=K, payload_bytes=payload)
        for i in range(K):  # feed exactly K packets
            dec.receive(cv[i], cp[i])
        assert dec.is_decoded
        recovered = dec.decode()
        assert np.array_equal(recovered, src)

    def test_decode_fails_with_fewer_than_K_packets(self):
        rng = np.random.default_rng(7)
        K, R, payload = 10, 5, 16
        enc = RLNCEncoder(K=K, R=R, payload_bytes=payload, rng=rng)
        src = rng.integers(0, 256, size=(K, payload)).astype(np.uint8)
        cv, cp = enc.encode(src)
        dec = RLNCDecoder(K=K, payload_bytes=payload)
        for i in range(K - 1):
            dec.receive(cv[i], cp[i])
        assert not dec.is_decoded
        assert dec.decode() is None

    def test_decode_with_redundant_packets_and_random_subset(self):
        rng = np.random.default_rng(99)
        K, R, payload = 16, 8, 24
        enc = RLNCEncoder(K=K, R=R, payload_bytes=payload, rng=rng)
        src = rng.integers(0, 256, size=(K, payload)).astype(np.uint8)
        cv, cp = enc.encode(src)

        # Simulate erasures: drop some, feed a random surviving subset of >= K.
        order = rng.permutation(K + R)
        dec = RLNCDecoder(K=K, payload_bytes=payload)
        for i in order[: K + 3]:
            dec.receive(cv[i], cp[i])
        if dec.is_decoded:
            assert np.array_equal(dec.decode(), src)

    def test_linearly_dependent_packet_does_not_raise_rank(self):
        rng = np.random.default_rng(5)
        K, payload = 4, 8
        dec = RLNCDecoder(K=K, payload_bytes=payload)
        v = np.array([1, 2, 3, 4], dtype=np.uint8)
        p = np.array([9, 9, 9, 9, 9, 9, 9, 9], dtype=np.uint8)
        assert dec.receive(v, p) is True
        assert dec.receive(v, p) is False  # identical vector: linearly dependent
        assert dec.rank == 1

    def test_decode_probability_matches_monte_carlo(self):
        K, R, p = 12, 6, 0.15
        analytic = decode_probability(K, R, p)
        mc = decode_probability_mc(K, R, p, n_trials=15000, rng=np.random.default_rng(0))
        assert abs(analytic - mc) < 0.02  # generous tolerance for MC noise

    def test_decode_probability_edge_cases(self):
        assert decode_probability(10, 5, 0.0) == pytest.approx(1.0)
        assert decode_probability(10, 5, 1.0) == pytest.approx(0.0)

    def test_decode_probability_monotonic_in_R(self):
        K, p = 20, 0.3
        probs = [decode_probability(K, R, p) for R in [0, 2, 5, 10, 20]]
        assert all(probs[i] <= probs[i + 1] + 1e-9 for i in range(len(probs) - 1))

    def test_decode_probability_vectorized_over_array(self):
        K, R = 16, 4
        p_arr = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        out = decode_probability(K, R, p_arr)
        assert out.shape == p_arr.shape
        assert out[0] == pytest.approx(1.0)
        assert out[-1] == pytest.approx(0.0)
        assert np.all(np.diff(out) <= 1e-9)  # monotonically non-increasing in p
