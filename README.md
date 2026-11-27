# python-data-analysis

Performance analysis and simulation for the KSU FSAE electric cars. The main piece is
an **acceleration event sim** (`accel/`) with wheelspin dynamics, a C++ launch /
traction controller (`controller/`) that's meant to run unchanged on the VCU, and
tooling to close the loop with car logs. The repo root also has older notebooks and
scripts for log analysis, motor curves and battery sims.

- [Setup](#setup)
  - [NixOS (or any machine with Nix)](#nixos-or-any-machine-with-nix)
  - [Without Nix (Linux, macOS, Windows/WSL)](#without-nix-linux-macos-windowswsl)
- [Quick start](#quick-start)
- [Accel sim](#accel-sim)
- [VCU controller (sim + car)](#vcu-controller-sim--car)
- [Visualization](#visualization)
- [Repo layout](#repo-layout)

## Setup

Python dependencies live in `pyproject.toml`, locked in `uv.lock`. Both setups below
install exactly those versions: with Nix, [uv2nix](https://github.com/pyproject-nix/uv2nix)
builds the environment from `uv.lock`; without Nix, [uv](https://docs.astral.sh/uv/) does.

### NixOS (or any machine with Nix)

Everything (Python, the locked packages, compilers, the Rerun viewer) comes from
`flake.nix`. No pip, no venv to manage.

1. Flakes need to be enabled. On NixOS, in your system config:
   ```nix
   nix.settings.experimental-features = [ "nix-command" "flakes" ];
   ```
   On other Linux/macOS, install Nix (https://nixos.org/download) and add
   `experimental-features = nix-command flakes` to `~/.config/nix/nix.conf`.
2. Clone and enter the dev shell:
   ```sh
   git clone https://github.com/mathbrook/python-data-analysis
   cd python-data-analysis
   nix develop            # first run downloads ~1-2 GB, then it's instant
   ```
   With [direnv](https://direnv.net), run `direnv allow` once and the shell loads
   whenever you `cd` in.
3. Check it works: `pytest -q` (about 10 s).

The shell also sets `PYTHONPATH` to the repo root and configures the `nbstripout`
git filter (below). `uv run ...` works inside it too and uses the Nix-built
environment (it never syncs or downloads). For the Teensy cross-compiler, use the
separate `nix develop .#firmware` shell (~1 GB, so it's not in the default one).

**Adding or upgrading a Python package:** `uv add <pkg>` (or `uv lock --upgrade-package
<pkg>`) inside the shell updates `pyproject.toml` and `uv.lock`; then re-enter the shell
(`exit` + `nix develop`, or `direnv reload`) and Nix rebuilds the environment from the
new lock. If a new package ships a wheel that links a system library it doesn't bundle,
Nix's build says which one; add it in `wheelLibs` in `flake.nix` (like `rerun-sdk` and
libudev). `rerun-sdk` is pinned to the Rerun viewer version nixpkgs ships, since the two
must match; bump them together.

### Without Nix (Linux, macOS, Windows/WSL)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), git, make and a
   C++17 compiler (uv fetches a suitable Python itself if you don't have one):
   - Debian/Ubuntu: `sudo apt install git build-essential` and
     `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - macOS: `xcode-select --install` and `brew install uv`
   - Windows: use WSL (Ubuntu) and follow the Ubuntu line. The controller builds a
     shared library with make/g++, which native Windows doesn't have.
2. Clone and sync (creates `.venv` from `uv.lock`, a few seconds):
   ```sh
   git clone https://github.com/mathbrook/python-data-analysis
   cd python-data-analysis
   uv sync
   ```
3. Set up the notebook output filter (commits are refused without it, see below):
   ```sh
   git config filter.nbstripout.clean "uv run --quiet nbstripout"
   git config filter.nbstripout.smudge cat
   git config filter.nbstripout.required true
   git config diff.ipynb.textconv "uv run --quiet nbstripout -t"
   ```
4. Check it works: `uv run pytest -q`.

Prefix commands with `uv run` (as in the quick start below), or `source .venv/bin/activate`
once per terminal and drop the prefix. Run them from the repo root: modules are imported
from there. Add packages with `uv add <pkg>`.

For the Teensy cross-compile, install the ARM toolchain
(`sudo apt install gcc-arm-none-eabi`, or Arm's GNU toolchain on macOS).

### Notebook outputs

Notebook outputs are stripped on commit by `nbstripout` (`.gitattributes`), so diffs
stay readable. Your local copies keep their outputs. The filter is marked required:
if `nbstripout` isn't set up, `git add` of a notebook fails rather than committing
megabytes of plots.

## Quick start

From the Nix dev shell, or without Nix prefix each command with `uv run`
(e.g. `uv run python -m accel ...`):

```sh
# accel times for a few final drives: ideal TC vs no TC vs TC
python -m accel --fd 2.4 3.0 4.0

# same, saving speed / accel / torque / slip traces
python -m accel --fd 3.0 --plot run.png

# 3D drag race replay of no TC vs the VCU controllers (opens the Rerun viewer)
python -m accel.viz

# the main notebook: traces, KS5e / OptimumLap validation, launch strategies,
# the C++ controller in the loop, Monte Carlo, the log -> fit -> retune loop
jupyter lab accel_sim.ipynb

# tests (sim physics, controller, logging/replay, system ID)
pytest -q
```

From Python:

```python
from accel import Car, MotorMap, SlipTC, simulate, quasi_static
from accel.firmware import FirmwareController

car = Car(final_drive=3.0)                 # defaults: 228HV, COSMX 144s1p, LC0 tire
ideal = quasi_static(car)                  # perfect-traction bound
run = simulate(car, controller=FirmwareController(mode="mu_adaptive"))
print(run.accel_time, ideal.accel_time)    # timed 75 m run, seconds
run.df                                     # every sim step: speed, slip, torque, loads, ...
run.controller_log                         # what the controller saw and commanded
```

## Accel sim

The `accel/` package is the acceleration event sim; `accel_sim.ipynb` is the plotting
front-end (`accel_weight_transfer.ipynb` is the old version, kept for reference).
Defaults are the current car: EMRAX 228HV, COSMX 144s1p pack, final drive 3.0, TTC
LC0 Pacejka tire.

- `params.py`: car parameters (mass, CG, aero, inertias, ...). Values marked ESTIMATE need measuring.
- `powertrain.py`: tabulated EMRAX torque(rpm, bus V) from `motor.py`, accumulator as Voc + R.
- `tire.py`: `PacejkaTire` (TTC fits in `tire/`, default LC0) or `CurveTire` (tire/mu-vs-slip.csv shape, scaled).
- `control.py`: slip-target PI traction control (Python, ideal sensors).
- `launch.py`: launch torque strategies (driver ramp, grip-limited launch map, numerical optimizer).
- `sim.py`: `simulate` (wheel spin dynamics) and `quasi_static` (ideal-TC bound, the joe-accel model).
- `compare.py`: reference data (OptimumLap run, KS5e launch logs) and matching sim setups;
  the validation plots are in `accel_sim.ipynb`.

## VCU controller (sim + car)

`controller/` is the launch/traction controller as firmware-style C++ (no heap, no Arduino, time passed
in, C ABI in `vcu_control.h`). The sim builds it with `make -C controller` (automatically, on first use)
and runs it through `accel/firmware.py`, feeding it modeled sensors: 19-tooth wheel speed edges, motor
rpm, torque feedback and VectorNav frames at CAN rates with latency and noise. The same sources are
meant to compile into the VCU.

- `accel/firmware.py`: `FirmwareController` (pass as `simulate(controller=...)`), input recording and
  `replay()` for running the controller on logged inputs.
- `accel/uncertainty.py`: Monte Carlo where the simulated car's coefficients are drawn from ranges while
  controllers keep their nominal beliefs.
- Calibration: the controller learns tire radii (front sensors vs VectorNav, rear motor vs ground speed
  while coasting), VectorNav accel bias (while staged) and grip; persist `vcu_learned_t` across runs.
  `accel.firmware.warmup()` simulates the calibration drive.
- Plausibility: the controller measures the gear ratio as motor revs over rear wheel revs
  (counted on the rear sensors' teeth, so slip and tire radius don't matter). If it disagrees
  with the configured `final_drive` by more than 8% twice in a row, it latches
  `VCU_FAULT_GEAR_RATIO` and passes the driver's torque straight through: with the wrong
  ratio, slip reads wrong and TC can hold the car on the line. A wrong ratio is caught
  within ~6 m of rolling, so the warmup drive to the line should catch it before a launch.
- Logging: `vcu_log_record_t` (packed binary, `vcu_control.h`) is what the car writes; `write_log` /
  `read_log` / `replay` in `accel/firmware.py`.
- `accel/sysid.py`: fits grip, mass, tire radius, rear inertia and tire relaxation from a log and narrows
  `uncertainty.RANGES`.
- `make -C controller teensy` (in `nix develop .#firmware`) cross-compiles the controller for the
  Teensy 4.1. CI (`.github/workflows/ci.yml`) runs the tests, the notebook and that cross-compile.

## Visualization

`python -m accel.viz` replays no TC vs the VCU controllers as a 3D drag race in [Rerun](https://rerun.io)
(one lane per run, body pitch from load transfer, spoked wheels so rear wheelspin is visible, arrows for rear
tire force and load) with speed / accel / slip / torque / TC / learned mu plots on the same timeline.
`--save runs.rrd` writes the recording instead; open it later with `rerun runs.rrd`. From code,
`accel.viz.show_runs({"name": result, ...}, car)`.

`python -m accel.report --out report/` runs the headline sims and writes a drag race GIF
(rear wheels colored by slip), launch / final drive / robustness / controller-vs-truth /
system ID / gear ratio check plots, a `summary.md` and a `metrics.json`. CI runs it on
every push: the tables and charts are on the run's summary page, and the GIF and plots are
in the `accel-report` artifact. On pull requests CI also runs the base branch's report and
posts `python -m accel.report_diff base/metrics.json report/metrics.json` as a PR comment
(accel times, robustness, system ID error, ... better / worse than base), updated on each push.

The Rerun viewer sends anonymous usage stats by default (features used, not your data);
`rerun analytics disable` turns that off.

## Repo layout

| Path | What |
|---|---|
| `accel/` | accel event sim, controller bridge, Monte Carlo, system ID, visualization |
| `controller/` | VCU launch/traction controller (C++), Makefile for sim and Teensy builds |
| `tests/` | pytest suite |
| `accel_sim.ipynb` | main accel notebook |
| `motor.py`, `battery.py`, `tire/` | motor dq model, pack model, Pacejka tire fits (used by `accel/`) |
| `cardata/`, `olapsims/`, `dynoruns/` | logged car data, OptimumLap export, dyno pulls |
| other `*.ipynb`, `*.py` | older log analysis, motor curve, battery and wiring scripts; many read logs from paths that aren't in the repo |
| `pyproject.toml`, `uv.lock` | Python dependencies and their lock (uv) |
| `flake.nix`, `flake.lock` | Nix dev shell: builds the env from `uv.lock` via uv2nix, plus compilers and the Rerun viewer |
