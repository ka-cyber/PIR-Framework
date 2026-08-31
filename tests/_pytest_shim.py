"""
Minimal, dependency-free stand-in for the subset of pytest's API this test
suite uses (`pytest.approx`, `pytest.raises`, `pytest.fixture`). This file
exists ONLY so the test suite can be verified in environments without
network access to `pip install pytest`; the intended, fully-supported way
to run these tests is the real pytest: `pip install pytest && pytest tests/`.
`tests/run_tests.py` uses this shim automatically if the real pytest isn't
importable.
"""
from __future__ import annotations

import math


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        if self.expected is None or other is None:
            return self.expected == other
        return math.isclose(other, self.expected, rel_tol=self.rel, abs_tol=self.abs)

    def __repr__(self):
        return f"approx({self.expected!r})"


def approx(expected, rel=1e-6, abs=1e-12):
    return _Approx(expected, rel=rel, abs=abs)


class raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type} to be raised, but nothing was raised")
        if not issubclass(exc_type, self.exc_type):
            return False
        return True


def fixture(*args, **kwargs):
    """No-op decorator: our runner calls fixture-producing methods manually."""
    def decorator(fn):
        fn._is_fixture = True
        return fn
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
    return decorator


def mark_skip(*args, **kwargs):
    def decorator(fn):
        fn._skip = True
        return fn
    return decorator


class _Mark:
    skip = staticmethod(mark_skip)


mark = _Mark()
