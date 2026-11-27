"""Accel event comparison for a few final drives: ideal TC vs no TC vs TC
(`accel sim --fd 3.0 4.0 --plot out.png`)."""

import pandas as pd

from kmslap import Car, CurveTire, MotorMap, SlipTC, quasi_static, simulate
from kmslap.car.tire import default_tire


def run(fds=(Car.final_drive,), motor: str = "228HV", pack: str = "cosmx600v", tire: str = "lc0",
        mu: float = 1.4, mass: float = Car.mass, plot_path: str | None = None):
    tire = default_tire() if tire == "lc0" else CurveTire(mu_peak=mu)
    motor = MotorMap(motor)
    rows, first = [], None
    for fd in fds:
        car = Car(final_drive=fd, mass=mass).open_aero()   # accel: wing open
        runs = [quasi_static(car, motor, pack, tire),
                simulate(car, motor, pack, tire),
                simulate(car, motor, pack, tire, tc=SlipTC.for_tire(tire))]
        first = first or runs
        rows += [{"fd": fd, **r.summary()} for r in runs]
    print(pd.DataFrame(rows).to_string(index=False))
    if plot_path:
        plot(first, plot_path)


def plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for r in runs:
        d = r.df
        axes[0].plot(d.t, d.v * 3.6, label=r.label)
        axes[1].plot(d.t, d.a / 9.81, label=r.label)
        axes[2].plot(d.t, d.motor_torque, label=r.label)
        if "tc_scale" in d:
            axes[3].plot(d.t, d.slip.clip(upper=1.0), label=f"{r.label} slip")
    for ax, ylabel in zip(axes, ["speed (km/h)", "accel (g)", "motor torque (Nm)", "slip ratio"]):
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print(f"saved {path}")

