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
- `tire.py`: `CurveTire` (tire/mu-vs-slip.csv shape, scaled) or `PacejkaTire` (TTC fits in `tire/`).
- `control.py`: slip-target PI traction control.
- `sim.py`: `simulate` (wheel spin dynamics) and `quasi_static` (ideal-TC bound, the joe-accel model).
- `compare.py`: reference data (OptimumLap run, KS5e launch logs) and matching sim setups;
  the validation plots are at the bottom of `accel_sim.ipynb`.
