# python-data-analysis

scripts to evaluate performance

## Setup (Nix)

```sh
nix develop            # or `direnv allow` once, then just cd in
jupyter lab            # notebooks
python -m motor        # scripts run from repo root (PYTHONPATH is set to it)
```

Python deps come from nixpkgs (see `flake.nix`), so there is no pip/venv step.
`requirements.txt` is kept for reference on non-Nix machines.

Notebook outputs are stripped on commit by `nbstripout` (configured automatically
by the dev shell; see `.gitattributes`). Your local copies keep their outputs.

## Accel sim

The `accel/` package is the acceleration event sim; `accel_sim.ipynb` is the plotting
front-end (`accel_weight_transfer.ipynb` is the old version, kept for reference).

```sh
python -m accel --fd 2.4 3.5 4.5            # ideal TC vs no TC vs TC per final drive
python -m accel --tire lc0 --plot run.png    # TTC Pacejka LC0 tire, save traces
pytest tests                                 # physics/TC sanity checks
```

- `params.py`: car parameters (mass, CG, aero, inertias, ...). Values marked ESTIMATE need measuring.
- `powertrain.py`: tabulated EMRAX torque(rpm, bus V) from `motor.py`, accumulator as Voc + R.
- `tire.py`: `PacejkaTire` (TTC fits in `tire/`, default LC0) or `CurveTire` (tire/mu-vs-slip.csv shape, scaled).
- `control.py`: slip-target PI traction control.
- `launch.py`: launch torque strategies (driver ramp, grip-limited launch map, numerical optimizer).
- `sim.py`: `simulate` (wheel spin dynamics) and `quasi_static` (ideal-TC bound, the joe-accel model).
- `compare.py`: reference data (OptimumLap run, KS5e launch logs) and matching sim setups;
  the validation plots are at the bottom of `accel_sim.ipynb`.

## VCU controller (sim + car)

`controller/` is the launch/traction controller as firmware-style C++ (no heap, no Arduino, time passed
in, C ABI in `vcu_control.h`). The sim builds it with `make -C controller` and runs it through
`accel/firmware.py`, feeding it modeled sensors: 19-tooth wheel speed edges, motor rpm and VectorNav
frames at CAN rates with latency and noise. The same sources are meant to compile into the VCU.

- `accel/firmware.py`: `FirmwareController` (pass as `simulate(controller=...)`), input recording and
  `replay()` for running the controller on logged inputs.
- `accel/uncertainty.py`: Monte Carlo where the simulated car's coefficients are drawn from ranges while
  controllers keep their nominal beliefs.
- Calibration: the controller learns tire radii (front sensors vs VectorNav, rear motor vs ground speed
  while coasting), VectorNav accel bias (while staged) and grip; persist `vcu_learned_t` across runs.
  `accel.firmware.warmup()` simulates the calibration drive.
- Logging: `vcu_log_record_t` (packed binary, `vcu_control.h`) is what the car writes; `write_log` /
  `read_log` / `replay` in `accel/firmware.py`.
- `accel/sysid.py`: fits grip, mass, tire radius, rear inertia and tire relaxation from a log and narrows
  `uncertainty.RANGES`.
- `make -C controller teensy` (in `nix develop .#firmware`) cross-compiles the controller for the
  Teensy 4.1. CI (`.github/workflows/ci.yml`) runs the tests, the notebook and that cross-compile.
