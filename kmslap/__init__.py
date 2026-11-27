"""kmslap: Kennesaw Motorsports lap sim. The car (car/), its models (models/), events
(events/), drivers (driving/), the VCU controller bridge (vcu/), strategy tuning,
validation, reporting and the tooling to run it all (ops/, cli).

`from kmslap import Car, simulate, ...` exports load on first use (PEP 562), so
`import kmslap` (and the CLI's tab completion) stays fast."""
import importlib

_EXPORTS = {
    "SlipTC": "vcu.control", "Car": "car.params", "MotorMap": "car.powertrain",
    "Pack": "car.powertrain", "PACKS": "car.powertrain", "Result": "events.accel",
    "quasi_static": "events.accel", "simulate": "events.accel", "CurveTire": "car.tire",
    "PacejkaTire": "car.tire", "default_tire": "car.tire",
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
