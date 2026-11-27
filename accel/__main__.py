"""Run the accel event: `python -m accel --fd 3.5 --tire lc0 --plot out.png`"""
import argparse

import pandas as pd

from . import Car, CurveTire, MotorMap, PACKS, PacejkaTire, SlipTC, quasi_static, simulate
from .powertrain import MOTORS
from .tire import default_tire


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fd", type=float, nargs="+", default=[Car.final_drive], help="final drive(s)")
    p.add_argument("--motor", choices=sorted(MOTORS), default="228HV")
    p.add_argument("--pack", choices=sorted(PACKS), default="cosmx600v")
    p.add_argument("--tire", choices=["curve", "lc0"], default="lc0")
    p.add_argument("--mu", type=float, default=1.4, help="peak mu for --tire curve")
    p.add_argument("--mass", type=float, default=Car.mass)
    p.add_argument("--plot", metavar="PNG", help="save time traces of the first final drive")
    args = p.parse_args()

    tire = default_tire() if args.tire == "lc0" else CurveTire(mu_peak=args.mu)
    motor = MotorMap(args.motor)
    rows, first = [], None
    for fd in args.fd:
        car = Car(final_drive=fd, mass=args.mass).open_aero()   # accel: wing open
        runs = [quasi_static(car, motor, args.pack, tire),
                simulate(car, motor, args.pack, tire),
                simulate(car, motor, args.pack, tire, tc=SlipTC.for_tire(tire))]
        first = first or runs
        rows += [{"fd": fd, **r.summary()} for r in runs]
    print(pd.DataFrame(rows).to_string(index=False))
    if args.plot:
        plot(first, args.plot)


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


if __name__ == "__main__":
    main()
