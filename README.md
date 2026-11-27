# kmslap

**kmslap** (Kennesaw Motorsports lap sim): performance simulation for the KSU FSAE
electric cars. A CasADi vehicle model of the car (1D for accel, four-wheel for
everything else) with its motor, pack (charge, heat) and tires; the events (accel,
skidpad, endurance on the real Michigan line); simulated drivers, DRS and regen; a C++
launch / traction controller (`controller/`) meant to run unchanged on the VCU, in the
loop; strategy tuning; and tooling to run it all on more than one machine. The older
log analysis notebooks and scripts are in `legacy/`.

| Package | What |
|---|---|
| `kmslap/car/` | `Car` parameters, tires (`PacejkaTire`, combined slip), motor map and pack (`powertrain.py`), and the original motor dq / battery / Pacejka fit models they're built on |
| `kmslap/models/` | CasADi models: `accel1d.py` (accel event plant), `twotrack.py` (four-wheel car), `envelope.py` (grip envelope by optimization) |
| `kmslap/events/` | `accel.py` (`simulate`, `quasi_static`), `launch.py` (launch strategies), `endurance.py`, `track.py` (real track lines) |
| `kmslap/driving/` | `paths.py` (lines, skidpad), `follower.py` (drivers), `policies.py` (DRS, regen), `drive.py` (the loop) |
| `kmslap/vcu/` | the C++ controller in the loop (`firmware.py`), Python TC, system ID, Monte Carlo over uncertain coefficients |
| `kmslap/strategy.py` | endurance strategy tuning (pace, coasting, DRS, regen) |
| `kmslap/validation/`, `reporting/` | KS5e / OptimumLap comparisons; the CI report, PR diff, Rerun 3D view |
| `kmslap/ops/` | remote runner, cluster across machines, benchmarks |
| `kmslap/cli.py` | the `kmslap` command |

- [Setup](#setup)
  - [NixOS (or any machine with Nix)](#nixos-or-any-machine-with-nix)
  - [Without Nix (Linux, macOS, Windows/WSL)](#without-nix-linux-macos-windowswsl)
- [Quick start](#quick-start)
- [Accel event](#accel-event)
- [Driving the car (double-track model)](#driving-the-car-double-track-model)
- [Endurance and DRS](#endurance-and-drs)
- [VCU controller (sim + car)](#vcu-controller-sim--car)
- [Running jobs on another machine](#running-jobs-on-another-machine)
- [Validation against car data](#validation-against-car-data)
- [Car data to collect](#car-data-to-collect)
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

Everything runs through one command, `kmslap`, with tab completion for commands, options,
choices (motors, packs) and Tailscale hosts. It's on PATH in the Nix dev shell (an
editable install of this repo, so code edits apply immediately). Without Nix, `uv sync`
installs it into `.venv/bin`: activate the venv to use it as `kmslap`, or prefix with
`uv run`. Turn completion on once:

```sh
nix profile install .#completions   # Nix: zsh and bash, no rc file edits (works with a
                                    # read-only home-manager .zshrc); new shells pick it up
kmslap --install-completion         # without Nix: appends to your shell's rc file
kmslap --help                       # commands: accel report report-diff viz drive endurance
                                    # strategy bench dyno pack-temps remote cluster
```

The completion only asks `kmslap` when you press TAB, so it's inert outside this repo's
shell; `nix profile remove completions` takes it out. `python -m kmslap ...` is the same
command.

```sh
# accel times for a few final drives: ideal TC vs no TC vs TC
kmslap accel 2.4 3.0 4.0

# same, saving speed / accel / torque / slip traces
kmslap accel 3.0 --plot run.png

# 3D drag race replay of no TC vs the VCU controllers (opens the Rerun viewer)
kmslap viz

# the main notebook: traces, KS5e / OptimumLap validation, launch strategies,
# the C++ controller in the loop, Monte Carlo, the log -> fit -> retune loop
jupyter lab accel_sim.ipynb

# tests (sim physics, controller, logging/replay, system ID)
pytest -q
```

From Python:

```python
from kmslap import Car, MotorMap, SlipTC, simulate, quasi_static
from kmslap.vcu.firmware import FirmwareController

car = Car(final_drive=3.0)                 # defaults: 228HV, COSMX 144s1p, LC0 tire
ideal = quasi_static(car)                  # perfect-traction bound
run = simulate(car, controller=FirmwareController(mode="mu_adaptive"))
print(run.accel_time, ideal.accel_time)    # timed 75 m run, seconds
run.df                                     # every sim step: speed, slip, torque, loads, ...
run.controller_log                         # what the controller saw and commanded
```

## Accel event

`kmslap/events/accel.py` is the acceleration event sim; `accel_sim.ipynb` is the plotting
front-end (`legacy/accel_weight_transfer.ipynb` is the old version, kept for reference).
Defaults are the current car: EMRAX 228HV, COSMX 144s1p pack, final drive 3.0, TTC
LC0 Pacejka tire.

- `car/params.py`: car parameters (mass, CG, aero, inertias, ...). Values marked ESTIMATE need measuring.
- `car/powertrain.py`: tabulated EMRAX torque(rpm, bus V) from `car/motor.py`, accumulator as Voc + R.
- `car/tire.py`: `PacejkaTire` (TTC fits in `car/pacejka/`, default LC0) or `CurveTire` (`car/pacejka/mu-vs-slip.csv` shape, scaled).
- `vcu/control.py`: slip-target PI traction control (Python, ideal sensors).
- `events/launch.py`: launch torque strategies (driver ramp, grip-limited launch map, numerical optimizer).
- `models/accel1d.py`: the plant (chassis, rear wheel spin, tire relaxation, load transfer lag, inverter
  lag, motor map, pack) as one [CasADi](https://web.casadi.org) model. `simulate` runs it as
  generated C (compiled on first use into `build/casadi/`, ~1 s); `rollout()` unrolls it for
  exact derivatives, e.g. `accel_time_sensitivity()` gives d(accel time)/d(every coefficient)
  in one pass. Grip and every `Car` field are parameters, so one compiled model serves all
  Monte Carlo plants.
- `events/accel.py`: `simulate` (the loop around the model: driver request, TC or the VCU controller,
  logging) and `quasi_static` (ideal-TC bound, the joe-accel model).
- `validation/compare.py`: reference data (OptimumLap run, KS5e launch logs) and matching sim setups;
  the validation plots are in `accel_sim.ipynb`.

## Driving the car (double-track model)

`kmslap/models/twotrack.py` is the whole car in four wheels, written in CasADi and compiled the
same way as the accel model: chassis (x, y, yaw), four wheel spins with the rears through
a Drexler-style limited-slip diff (preload + locking torque proportional to input
torque), combined-slip Pacejka tires with relaxation lengths, longitudinal and lateral
load transfer (lagged), aero, brakes, the motor map, and the pack with its state of
charge and temperature (I²R heating, passive cooling to ambient). New `Car` and `Pack`
fields for it are marked ESTIMATE.

`kmslap/driving/drive.py` runs it from a driver's pedal, brake and steering wheel, optionally through
the VCU controller (fed per-wheel sensor edges):

```python
from kmslap.driving.drive import drive
from kmslap.driving.follower import PathFollower
from kmslap.driving.paths import skidpad
res = drive(PathFollower(skidpad(), speed=10.0))
res.df[["t", "X", "Y", "speed", "ay", "beta", "soc", "pack_temp"]]
```

`kmslap/driving/follower.py` has `ScriptedDriver` (inputs as functions of time) and `PathFollower` (pure
path tracking steering, acceleration-limited speed control, lifts when the rear slides);
`paths.py` the lines (`skidpad()`, `Path.from_segments`) and a speed profile from acceleration limits.
Target speeds come from the car itself: `kmslap/models/envelope.py` solves its grip envelope
(g-g-v diagram) with Ipopt on the same tire, load transfer, aero, diff and motor model:
at each speed and direction in the g-g plane, the steady state that maximizes
acceleration, required to be stable (no drifts the driver would have to catch).
`envelope.speed_profile(path, env)` turns it into a speed target along a path (90% of the
envelope by default: what the driver holds in steady cornering in the time-domain model).
The envelope takes ~2 minutes and is cached in `build/envelope/`.
`kmslap drive --out drive.png` plots a skidpad and a short course. With straight
steering it reproduces the accel model's times; the tests check steady cornering, left/right
symmetry, pack charge and heat bookkeeping and the diff.

## Endurance and DRS

`kmslap/events/endurance.py` runs the FSAE endurance event on the double-track car: the official
Michigan 2026 EV line (`tracks/`, from the fsae-ev-endurance-map repo; `kmslap/events/track.py`
loads and smooths it, ~1.22 km), 18 laps in two stints with the 3 minute driver change
(the pack cools toward ambient meanwhile), driven at a fraction of the grip envelope.
The pack discharges and heats (I²R, passive cooling only) and its BMS undervoltage limit
cuts power as it empties, so a car that uses too much energy slows and stops short.

```sh
kmslap endurance --out endurance.png
# wing closed, wing set open by hand, DRS, wing closed + lift-and-coast
```

The summary has driving time, lap times, energy (net, and put back by regen), drag
energy, final SOC, pack temperature and heat, and whether it finished. `PathFollower(coast_time=...)` is
lift-and-coast (lift when a braking zone is coming). `Car()` is the current car with
its wing closed; `Car().open_aero()` is the wing set open by hand (the accel tooling
uses that). DRS switches between the two while driving: a model input (wing position
blending drag, downforce and aero balance) driven by a `DrsPolicy` (open when fast,
nearly straight, on the gas, not braking; actuator travel time). Aero drag is from
python-endurance-drag-energy-calculator's calc_drag.ipynb; downforce is a placeholder
(see below).

**Regen** is off unless `drive()` / `endurance()` get a `RegenPolicy`; off, the model is
exactly the motoring-only one. On, negative motor torque generates through the rear
wheels (diff coast side, chain, the EMRAX efficiency map, inverter, cell coulombic
efficiency), limited by the policy's torque setting, the motor map, a 5 km/h minimum
speed (fading in to 8 km/h), and the pack's continuous charge current and cell max
voltage from the COSMX datasheet. Two strategies, combinable: `lift_torque` puts regen
at the bottom of the pedal map (one-pedal style: continuous, so the driver's speed
control can't chatter between drive and regen), and `brake_fraction` blends regen on
top of the friction brakes (from the front/rear pressure sensors), capped by
`max_rear_brake_share` since regen is rear-only. Brake bias is a static `Car` setting.
The C++ controller only handles drive torque so far: regen commands go straight to the
inverter.

**Strategy tuning:** `kmslap strategy --out strategy/ [--regen]` searches pace
(fraction of the grip envelope), lift-and-coast, DRS policy parameters and (with
`--regen`) regen strategies for the fastest event that still finishes with a 10% SOC
reserve. Candidates are scored on 2 laps projected to the event (energy from the cells
through the pack's voltage curve; checked against full events to ~0.3% SOC), in
parallel, and the winners confirmed with full 18-lap events.

## VCU controller (sim + car)

`controller/` is the launch/traction controller as firmware-style C++ (no heap, no Arduino, time passed
in, C ABI in `vcu_control.h`). The sim builds it with `make -C controller` (automatically, on first use)
and runs it through `kmslap/vcu/firmware.py`, feeding it modeled sensors: 19-tooth wheel speed edges, motor
rpm, torque feedback and VectorNav frames at CAN rates with latency and noise. The same sources are
meant to compile into the VCU.

- `kmslap/vcu/firmware.py`: `FirmwareController` (pass as `simulate(controller=...)`), input recording and
  `replay()` for running the controller on logged inputs.
- `kmslap/vcu/uncertainty.py`: Monte Carlo where the simulated car's coefficients are drawn from ranges while
  controllers keep their nominal beliefs.
- Calibration: the controller learns tire radii (front sensors vs VectorNav, rear motor vs ground speed
  while coasting), VectorNav accel bias (while staged) and grip; persist `vcu_learned_t` across runs.
  `kmslap.vcu.firmware.warmup()` simulates the calibration drive.
- Plausibility: the controller measures the gear ratio as motor revs over rear wheel revs
  (counted on the rear sensors' teeth, so slip and tire radius don't matter). If it disagrees
  with the configured `final_drive` by more than 8% twice in a row, it latches
  `VCU_FAULT_GEAR_RATIO` and passes the driver's torque straight through: with the wrong
  ratio, slip reads wrong and TC can hold the car on the line. A wrong ratio is caught
  within ~6 m of rolling, so the warmup drive to the line should catch it before a launch.
- Logging: `vcu_log_record_t` (packed binary, `vcu_control.h`) is what the car writes; `write_log` /
  `read_log` / `replay` in `kmslap/vcu/firmware.py`.
- `kmslap/vcu/sysid.py`: fits grip, mass, tire radius, rear inertia and tire relaxation from a log and narrows
  `uncertainty.RANGES`.
- `make -C controller teensy` (in `nix develop .#firmware`) cross-compiles the controller for the
  Teensy 4.1. CI (`.github/workflows/ci.yml`) runs the tests, the notebook and that cross-compile.

## Visualization

`kmslap viz` replays no TC vs the VCU controllers as a 3D drag race in [Rerun](https://rerun.io)
(one lane per run, body pitch from load transfer, spoked wheels so rear wheelspin is visible, arrows for rear
tire force and load) with speed / accel / slip / torque / TC / learned mu plots on the same timeline.
`--save runs.rrd` writes the recording instead; open it later with `rerun runs.rrd`. From code,
`kmslap.reporting.viz.show_runs({"name": result, ...}, car)`.

`kmslap report --out report/` runs the headline sims and writes a drag race GIF
(rear wheels colored by slip), launch / final drive / robustness / controller-vs-truth /
system ID / gear ratio check / sensitivity ("what to measure next": seconds of accel time
each uncertain coefficient is worth) plots, a `summary.md` and a `metrics.json`. CI runs it on
every push: the tables and charts are on the run's summary page, and the GIF and plots are
in the `accel-report` artifact. On pull requests CI also runs the base branch's report and
posts `kmslap report-diff base/metrics.json report/metrics.json` as a PR comment
(accel times, robustness, system ID error, ... better / worse than base), updated on each push.

The Rerun viewer sends anonymous usage stats by default (features used, not your data);
`rerun analytics disable` turns that off.

## Running jobs on another machine

`kmslap/ops/remote.py` runs a command on another machine (default the desktop's WSL,
`desktop-17id03o-2` on Tailscale) on exactly your working tree, and copies results back:

```sh
kmslap remote setup       # once: checks git + nix, clones, builds the env
kmslap remote run --fetch strategy -- kmslap strategy --regen --out strategy
kmslap remote run --detach --fetch strategy -- kmslap strategy --out strategy
kmslap remote log         # follow a detached job; then: fetch strategy
```

It snapshots tracked files including uncommitted edits (`git stash create`, your branch
and stash untouched), pushes that over SSH, checks it out there and runs it in
`nix develop` (`--uv` for `uv run`). Untracked files aren't sent. `--host` /
`$ACCEL_REMOTE_HOST` and `--dir` / `$ACCEL_REMOTE_DIR` pick another machine or clone.

The remote, once (Windows: WSL2 Ubuntu):

1. `wsl --install`, then in WSL: Tailscale (or use the Windows host's), an SSH server
   (`sudo apt install openssh-server && sudo service ssh start`), git, and Nix with
   flakes (e.g. the Determinate Systems installer).
2. From this machine: `ssh-copy-id <host>` so SSH works without a password (the tool
   runs SSH non-interactively), then `kmslap remote setup`.

Parallel jobs (`kmslap strategy`, Monte Carlo) use all the remote's cores
(`--workers`).

**Both machines at once:** `kmslap/ops/cluster.py` spreads independent tasks over this
machine and remotes: a task queue here (stdlib `multiprocessing.managers`), one worker
per core on each machine pulling from it (faster machines take more), remotes reached
through an SSH reverse tunnel with the working tree pushed as above. In a terminal it
shows a live dashboard (per-core load per machine, each worker's task and elapsed time,
progress, throughput, ETA); `save_timeline()` plots every task on every worker and the
CPU load.

```sh
kmslap strategy --regen --out strategy --remote mathewos@desktop-17id03o-2
kmslap cluster selftest --remote mathewos@desktop-17id03o-2   # try the dashboard
kmslap bench single|parallel|cluster --out bench/             # benchmarks
```

Benchmarks (Nov 2026): the laptop (Ryzen 5 7640U, 12 threads) runs single sims ~25%
faster than the desktop's WSL (Ryzen 7 5800XT, 16 threads); in parallel they're even,
1.33 vs 1.28 strategy evaluations/s; the cluster ran both at 2.41/s, 93% of their sum
(the rest is start-up and the last tasks finishing).

## Validation against car data

`kmslap/validation/` checks the models against the team's logs:

- `compare.py`: KS5e launches and an OptimumLap run against the accel sim (`accel_sim.ipynb`).
- `dyno.py` (`kmslap dyno`): dyno pulls (`cardata/dyno/`) against the inverter log of the
  same pull, aligned by speed, and against the motor model. On the KS6e 2023 pull the
  DC → wheels efficiency is 0.875 measured vs 0.870 from the model (motor map × inverter
  × driveline), the fitted roller radius at FD 3.0 is the sim's 0.229 m, and the inverter's
  torque feedback is within 5% of the dyno's. The 2024 228HV pull, on a pack sagging to
  ~176 V, is 16% less efficient than the map: deep field weakening the map doesn't cover.
- `pack_thermal.py` (`kmslap pack-temps`): the KS6e accumulator tests as top-down cell
  heatmaps (GIF) and the sim's lumped pack thermal model fitted to them (0.4 K RMS without
  fans: R ~1.3x the datasheet's, hA ~11.5 W/K for that pack; the with-fans test never
  cools down, so its hA isn't identified).

`legacy/README.md` reviews the old notebooks and scripts and indexes the team log
archive (endurance, aero tests, regen, more dyno and thermal data) for what to validate next.

## Car data to collect

TODO: everything the sim uses that hasn't been measured on the current car. Values in
`kmslap/car/params.py` (`Car`), `kmslap/car/powertrain.py` (`Pack`, motor) and `kmslap/vcu/firmware.py`
(sensors) are marked ESTIMATE / PLACEHOLDER / VERIFY in the code; `kmslap report`'s
"what to measure next" ranks how much each matters for accel time.

- **Aero, wing closed and open:** ClA and aero balance (only drag is ours, from
  calc_drag.ipynb). The downforce is a placeholder copied from other teams: closed ClA
  5.0 is our CdA times the L/D (2.4) of a published full-aero FSAE car
  ([ClA 5.47 / CdA 2.28](https://pidduuu.github.io/jekyll/update/2023/10/01/Justification-for-Aerodynamics.html));
  open ClA 3.0 applies [Monash's FSAE DRS](https://people.eng.unimelb.edu.au/imarusic/proceedings/19/387.pdf)
  downforce losses (front wing -37%, rear wing -67%) to a typical wing/undertray split.
  Coast-down tests in both configs would give CdA; ride height sensors or shock pots
  under speed give ClA and balance. DRS actuator travel time.
- **Mass properties:** mass with driver, CG height, weight distribution, yaw inertia,
  wheel and driveline inertias, motor rotor inertia (EMRAX datasheet, VERIFY).
- **Geometry:** front/rear track, steering ratio and maximum steering angle,
  loaded tire radius.
- **Suspension:** roll stiffness distribution, pitch/roll response times.
- **Diff:** Drexler preload and ramp angles (locking factors, drive and coast).
- **Brakes:** bias and maximum brake torque.
- **Tires:** road grip scale vs the TTC fit (0.6 now), relaxation lengths.
- **Pack:** COSMX cell open-circuit voltage curve (the repo's is adjusted from LG HE2,
  not measured, and tops out at 4.2 V where the datasheet allows 4.28), heat capacity,
  pack-to-ambient heat transfer (no cooling), resistance vs temperature and current (the
  sim uses the datasheet's 1.5 mOhm high-rate DCIR; it's up to 5 mOhm at low rates), the
  BMS's actual limits (the sim uses the datasheet's continuous ratings: 149.5 A
  discharge, 65 A charge, 3.0-4.28 V; the 2 s peaks aren't modeled). Datasheet values are
  in `datasheets/cosmx_95b0d0hd.toml`.
- **Powertrain:** inverter torque response, driveline efficiency, rolling resistance,
  inverter efficiency (regen uses 0.97), whether the inverter regens in field
  weakening, the inverter's torque limit (230 Nm in the sim; the EMRAX 228 datasheet
  allows 240 for a few seconds). The motor efficiency map is digitized from the EMRAX
  datasheet (motoring; assumed the same generating); outside its 86% contour it's a
  fitted loss model.
- **Sensors / CAN:** VectorNav breakout send rates and latency; INS status on CAN; the
  VCU will need the DRS wing state too (its grip model depends on the downforce).

## Repo layout

| Path | What |
|---|---|
| `kmslap/` | the package (see the table at the top) |
| `controller/` | VCU launch/traction controller (C++), Makefile for sim and Teensy builds |
| `build/` | generated, gitignored: the compiled CasADi model |
| `tests/` | pytest suite |
| `accel_sim.ipynb` | main accel notebook |
| `tracks/` | track lines (GeoJSON): FSAE Michigan 2026 EV endurance |
| `datasheets/` | COSMX cell (image + `cosmx_95b0d0hd.toml`, read by `Pack`), EMRAX 228 PDF, its efficiency map digitized to CSV (`extract_emrax_efficiency.py`, with a check overlay) |
| `cardata/`, `olapsims/` | logged car data (KS5e launches, KS6e accumulator tests, dyno pulls in `cardata/dyno/`), OptimumLap export |
| `legacy/` | older log analysis, motor curve, battery and wiring notebooks and scripts (many read logs from paths that aren't in the repo); to review for what's worth bringing back |
| `pyproject.toml`, `uv.lock` | Python dependencies and their lock (uv) |
| `flake.nix`, `flake.lock` | Nix dev shell: builds the env from `uv.lock` via uv2nix, plus compilers and the Rerun viewer |
