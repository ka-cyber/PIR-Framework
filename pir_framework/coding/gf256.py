"""
GF(256) = GF(2^8) finite-field arithmetic for Random Linear Network Coding.

Construction
------------
We use the Rijndael/Reed-Solomon-style primitive polynomial

    x^8 + x^4 + x^3 + x^2 + 1   (hex 0x11D)

with generator g = 2. Under this modulus, 2 is a primitive element (its
multiplicative order is 255, i.e. it generates the entire non-zero field),
which lets us build O(1)-multiplication log/antilog (Zech logarithm) tables
after a one-time O(256) setup. This is the same field construction used by
production RLNC/Reed-Solomon libraries such as `kodo` and `gf-complete`,
and matches the field used in the APC-RLNC reference implementation.

All operations are fully vectorized over NumPy arrays of dtype uint8, so
encoding/decoding an entire generation of packets is a small number of
array operations rather than a Python-level loop over bytes.
"""
from __future__ import annotations

import numpy as np

PRIMITIVE_POLY: int = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1

# --------------------------------------------------------------------------
# Build log / antilog tables once at import time (module-level singletons).
# --------------------------------------------------------------------------
_EXP = np.zeros(512, dtype=np.uint16)  # antilog table, doubled to skip a mod
_LOG = np.zeros(256, dtype=np.uint16)  # log table


def _build_tables() -> None:
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIMITIVE_POLY
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]
    _LOG[0] = 0  # log(0) is undefined; guarded explicitly below


_build_tables()


def gf_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Addition (and subtraction) in GF(2^8): bitwise XOR."""
    return np.bitwise_xor(np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8))


def gf_mul(a, b):
    """Elementwise multiplication in GF(256). Accepts scalars or ndarrays."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    shape = np.broadcast_shapes(a.shape, b.shape)
    a_b, b_b = np.broadcast_to(a, shape), np.broadcast_to(b, shape)
    out = np.zeros(shape, dtype=np.uint8)
    nz = (a_b != 0) & (b_b != 0)
    if np.any(nz):
        la = _LOG[a_b[nz]].astype(np.int64)
        lb = _LOG[b_b[nz]].astype(np.int64)
        out[nz] = _EXP[(la + lb) % 255].astype(np.uint8)
    if shape == ():
        return np.uint8(out)
    return out


def gf_inv(a) -> np.ndarray:
    """Multiplicative inverse in GF(256). gf_inv(0) raises ZeroDivisionError."""
    a = np.asarray(a, dtype=np.int64)
    if np.any(a == 0):
        raise ZeroDivisionError("GF(256) inverse of 0 is undefined")
    la = _LOG[a]
    inv_log = (255 - la) % 255
    out = _EXP[inv_log].astype(np.uint8)
    return out if out.shape else np.uint8(out)


def gf_div(a, b) -> np.ndarray:
    """Division a / b in GF(256). Division by zero raises ZeroDivisionError."""
    b = np.asarray(b, dtype=np.int64)
    if np.any(b == 0):
        raise ZeroDivisionError("GF(256) division by 0")
    return gf_mul(a, gf_inv(b))


def gf_pow(a, n: int) -> np.ndarray:
    """a ** n in GF(256) for non-negative integer exponent n."""
    a = np.asarray(a, dtype=np.int64)
    out = np.ones_like(a, dtype=np.uint8)
    base = a.copy()
    e = n
    while e > 0:
        if e & 1:
            out = gf_mul(out, base)
        base = gf_mul(base, base)
        e >>= 1
    return out if out.shape else np.uint8(out)


class GF256:
    """
    Thin object-oriented convenience wrapper around the module-level
    vectorized GF(256) functions, mainly used for RLNC coefficient-matrix
    linear algebra (Gaussian elimination).
    """

    add = staticmethod(gf_add)
    sub = staticmethod(gf_add)  # char-2 field: subtraction == addition
    mul = staticmethod(gf_mul)
    div = staticmethod(gf_div)
    inv = staticmethod(gf_inv)
    pow = staticmethod(gf_pow)

    @staticmethod
    def random_nonzero_matrix(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
        """Sample a `rows`x`cols` matrix of uniform *non-zero* GF(256) coefficients.

        RLNC coding vectors are drawn from GF(256)\\{0} in practice (using 0
        coefficients everywhere would waste degrees of freedom); this keeps
        coding matrices well-conditioned for the field sizes used here.
        """
        return rng.integers(1, 256, size=(rows, cols)).astype(np.uint8)

    @staticmethod
    def identity(n: int) -> np.ndarray:
        return np.eye(n, dtype=np.uint8)

    @staticmethod
    def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication over GF(256): C[i,j] = XOR_k GF_mul(A[i,k], B[k,j])."""
        A = np.asarray(A, dtype=np.uint8)
        B = np.asarray(B, dtype=np.uint8)
        assert A.shape[1] == B.shape[0], f"shape mismatch {A.shape} @ {B.shape}"
        # Broadcast-multiply then XOR-reduce along the shared axis.
        prod = gf_mul(A[:, :, None], B[None, :, :])  # (m, k, n)
        out = np.zeros((A.shape[0], B.shape[1]), dtype=np.uint8)
        for k in range(A.shape[1]):
            out ^= prod[:, k, :]
        return out

    @staticmethod
    def rank(matrix: np.ndarray) -> int:
        """Rank of a GF(256) matrix via Gaussian elimination (row echelon form)."""
        M = np.array(matrix, dtype=np.int64, copy=True)
        rows, cols = M.shape
        rank = 0
        for col in range(cols):
            pivot = None
            for r in range(rank, rows):
                if M[r, col] != 0:
                    pivot = r
                    break
            if pivot is None:
                continue
            M[[rank, pivot]] = M[[pivot, rank]]
            inv_p = int(gf_inv(np.uint8(M[rank, col])))
            M[rank, :] = gf_mul(M[rank, :], inv_p)
            for r in range(rows):
                if r != rank and M[r, col] != 0:
                    factor = M[r, col]
                    M[r, :] = gf_add(M[r, :], gf_mul(factor, M[rank, :]))
            rank += 1
            if rank == rows:
                break
        return rank
