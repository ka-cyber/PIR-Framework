"""
Test runner for the PIR-Framework test suite.

Preferred usage (full pytest, with fixtures, parametrization, etc.):
    pip install pytest
    pytest tests/

Fallback usage (no network access to install pytest -- what this script
does): a small reflection-based runner that discovers `Test*` classes and
`test_*` methods/functions in `tests/test_*.py`, injects `pytest` as the
local shim module (`_pytest_shim.py`) so `pytest.approx`/`pytest.raises`
still work, manually resolves any single no-argument class-scoped fixture
a test class declares, and reports pass/fail/error counts with tracebacks
for failures.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent


def _install_pytest_shim() -> None:
    try:
        import pytest  # noqa: F401
        return  # real pytest is available; nothing to do
    except ImportError:
        pass
    sys.path.insert(0, str(TESTS_DIR))
    import _pytest_shim
    sys.modules["pytest"] = _pytest_shim


def _resolve_fixtures(cls) -> dict:
    """For a Test* class, find methods decorated as fixtures (our shim just
    tags them with `_is_fixture`) and call them once (class-scoped
    semantics), returning {fixture_name: value}."""
    resolved = {}
    instance = cls()
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if getattr(method, "_is_fixture", False):
            resolved[name] = method(instance)
    return resolved


def run() -> int:
    _install_pytest_shim()
    sys.path.insert(0, str(REPO_ROOT))

    test_files = sorted(TESTS_DIR.glob("test_*.py"))
    n_pass, n_fail, n_error = 0, 0, 0
    failures = []

    for test_file in test_files:
        module_name = test_file.stem
        spec = importlib.util.spec_from_file_location(module_name, test_file)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            print(f"[ERROR loading module] {module_name}")
            traceback.print_exc()
            n_error += 1
            continue

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and name.startswith("Test"):
                fixtures = _resolve_fixtures(obj)
                instance = obj()
                for meth_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                    if not meth_name.startswith("test_"):
                        continue
                    if getattr(method, "_skip", False):
                        continue
                    sig = inspect.signature(method)
                    kwargs = {}
                    for param in list(sig.parameters)[1:]:  # skip 'self'
                        if param in fixtures:
                            kwargs[param] = fixtures[param]
                    full_name = f"{module_name}.{name}.{meth_name}"
                    try:
                        method(instance, **kwargs)
                        n_pass += 1
                    except Exception:
                        n_fail += 1
                        failures.append((full_name, traceback.format_exc()))
            elif inspect.isfunction(obj) and name.startswith("test_") and obj.__module__ == module_name:
                full_name = f"{module_name}.{name}"
                try:
                    obj()
                    n_pass += 1
                except Exception:
                    n_fail += 1
                    failures.append((full_name, traceback.format_exc()))

    print(f"\n{'=' * 70}\n{n_pass} passed, {n_fail} failed, {n_error} module load errors\n{'=' * 70}")
    for name, tb in failures:
        print(f"\n--- FAILED: {name} ---\n{tb}")

    return 0 if (n_fail == 0 and n_error == 0) else 1


if __name__ == "__main__":
    raise SystemExit(run())
