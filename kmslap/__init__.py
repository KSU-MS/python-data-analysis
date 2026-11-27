"""FSAE EV sims. `from accel import Car, simulate, ...` works as usual; the submodules
load on first use (PEP 562), so `import accel` (and the CLI's tab completion) stays
fast."""
import importlib

_EXPORTS = {
    "SlipTC": "control", "Car": "params", "MotorMap": "powertrain", "Pack": "powertrain",
    "PACKS": "powertrain", "Result": "sim", "quasi_static": "sim", "simulate": "sim",
    "CurveTire": "tire", "PacejkaTire": "tire", "default_tire": "tire",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_EXPORTS))
