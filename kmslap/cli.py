"""The `kmslap` command: every tool in this package as a subcommand, with tab completion.

    kmslap --install-completion     # once, for your shell (zsh / bash / fish)
    kmslap --help                   # or `kmslap <command> --help`

Installed by `uv sync` (non-Nix) or present in the Nix dev shell (editable install).
`python -m kmslap ...` is the same command.

Commands import what they need only when they run, so completion stays fast.
"""
import json
import shutil
import subprocess
from enum import Enum
from typing import Annotated, Optional

import typer

from kmslap.ops import remote as _remote_defaults  # light: stdlib only

app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich", add_completion=True,
                  help="FSAE sims: accel, the VCU controller, endurance, strategy tuning, and "
                       "running them on other machines.")
remote_app = typer.Typer(no_args_is_help=True, help="Run jobs on another machine (ops.remote).")
cluster_app = typer.Typer(no_args_is_help=True, help="Spread jobs over machines (ops.cluster).")
app.add_typer(remote_app, name="remote")
app.add_typer(cluster_app, name="cluster")


# --- choices (listed here so completion doesn't import the model; tests check them) ---

class Motor(str, Enum):
    m208HV = "208HV"
    m208MV = "208MV"
    m228HV = "228HV"
    m228MV = "228MV"


class PackName(str, Enum):
    cosmx300v = "cosmx300v"
    cosmx400v = "cosmx400v"
    cosmx600v = "cosmx600v"
    energus250v = "energus250v"
    energus300v = "energus300v"


class Tire(str, Enum):
    lc0 = "lc0"
    curve = "curve"


class TrackName(str, Enum):
    michigan2026 = "michigan2026"
    ac18 = "ac18"


class Bench(str, Enum):
    single = "single"
    parallel = "parallel"
    cluster = "cluster"


def complete_hosts(incomplete: str) -> list[str]:
    """Online Tailscale machines (and the default remote host) for --remote / --host."""
    hosts = [_remote_defaults.DEFAULT_HOST]
    if shutil.which("tailscale"):
        try:
            status = json.loads(subprocess.run(["tailscale", "status", "--json"], capture_output=True,
                                               text=True, timeout=2).stdout)
            for peer in (status.get("Peer") or {}).values():
                if peer.get("Online"):
                    hosts.append(peer.get("DNSName", "").split(".")[0] or peer.get("HostName", ""))
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
            pass
    user = _remote_defaults.DEFAULT_HOST.split("@")[0] + "@" if "@" in _remote_defaults.DEFAULT_HOST else ""
    out = []
    for h in hosts:
        for cand in (h, f"{user}{h}" if user and "@" not in h else h):
            if cand and cand.startswith(incomplete) and cand not in out:
                out.append(cand)
    return out


Remote = Annotated[list[str], typer.Option("--remote", "-r", autocompletion=complete_hosts,
                                           help="also run on this machine (repeatable)")]
Workers = Annotated[Optional[int], typer.Option(help="workers here (default: all cores)")]
Host = Annotated[str, typer.Option(autocompletion=complete_hosts, help="user@host to run on")]
Dir = Annotated[str, typer.Option("--dir", help="the remote clone, relative to its $HOME")]


# --- sims ---------------------------------------------------------------------------

@app.command()
def accel(fd: Annotated[Optional[list[float]], typer.Argument(help="final drive(s)", show_default="3.0")] = None,
        motor: Motor = Motor.m228HV, pack: PackName = PackName.cosmx600v, tire: Tire = Tire.lc0,
        mu: Annotated[float, typer.Option(help="peak mu for --tire curve")] = 1.4,
        mass: Annotated[Optional[float], typer.Option(help="kg (default: Car's)")] = None,
        plot: Annotated[Optional[str], typer.Option(help="save time traces (PNG)")] = None):
    """Accel event: ideal TC vs no TC vs TC, for each final drive."""
    from kmslap.car.params import Car
    from kmslap.events.accel_compare import run
    run(fd or [Car.final_drive], motor.value, pack.value, tire.value, mu,
        mass if mass is not None else Car.mass, plot)


@app.command()
def report(out: str = "report", plants: Annotated[int, typer.Option(help="Monte Carlo plants")] = 20):
    """Headline accel sims: drag race GIF, plots, summary.md, metrics.json (what CI runs)."""
    from kmslap.reporting.report import run
    run(out, plants)


@app.command("report-diff")
def report_diff(base: str, head: str, artifact_url: Optional[str] = None):
    """Compare two metrics.json files as markdown (the PR comment)."""
    from kmslap.reporting.report_diff import run
    run(base, head, artifact_url)


@app.command("junit-summary")
def junit_summary(xml: str, artifact_url: Optional[str] = None):
    """pytest's --junitxml results as a markdown table (the CI job summary)."""
    from kmslap.reporting.junit_summary import run
    run(xml, artifact_url)


@app.command()
def viz(save: Annotated[Optional[str], typer.Option(help="write this .rrd instead of opening the viewer")] = None,
        fd: float = 3.0):
    """3D drag race of no TC vs the VCU controllers in Rerun."""
    from kmslap.reporting.viz import run
    run(save, fd)


@app.command()
def drive(out: str = "drive.png"):
    """Double-track demo: skidpad and a short course, plotted."""
    from kmslap.driving.drive import demo
    demo(out)


@app.command()
def endurance(laps: int = 18, margin: Annotated[float, typer.Option(help="fraction of the grip envelope")] = 0.8,
              out: str = "endurance.png"):
    """Endurance event: wing closed / open / DRS / lift-and-coast compared."""
    from kmslap.events.endurance import compare
    compare(laps, margin, out)


@app.command()
def laptime(track: Annotated[TrackName, typer.Argument(help="michigan2026: the traced 2026 line, 4.5 m wide; "
                                                             "ac18: the Assetto Corsa cone course")] = TrackName.michigan2026,
            grip: Annotated[float, typer.Option(help="fraction of the tires' grip the plan uses")] = 0.85,
            margin: Annotated[float, typer.Option(help="m kept between the car's side and the edge / cones")] = 0.15,
            step: Annotated[float, typer.Option(help="m between the solver's nodes")] = 2.0,
            label: Annotated[Optional[str], typer.Option(help="name of the saved lap")] = None,
            width: Annotated[Optional[float], typer.Option(help="track width, m (michigan2026: a line only)")] = None,
            verbose: bool = False):
    """Minimum-time racing line (double track, ~5-7 min, cached), driven, scored, lap saved."""
    from kmslap.events.laptime import run
    run(track.value, grip, margin, step, label, verbose, width)


@app.command()
def race(track: Annotated[TrackName, typer.Argument()] = TrackName.michigan2026,
         label: Annotated[list[str], typer.Option(help="only these saved laps (repeatable)")] = [],
         out: Annotated[str, typer.Option(help=".gif, or .mp4 (smaller, faster)")] = "race.mp4",
         speedup: Annotated[float, typer.Option(help="playback speed")] = 4.0,
         rerun: Annotated[bool, typer.Option("--rerun", help="3D in the Rerun viewer instead")] = False,
         save: Annotated[Optional[str], typer.Option(help="with --rerun: write this .rrd, don't open")] = None):
    """Race the laps saved on a track (kmslap laptime, ...) against each other: a video, or 3D in Rerun."""
    from kmslap.events.track import TRACK_NAMES
    from kmslap.reporting import birdseye, runs
    laps = runs.load_laps(track.value, label or None)
    if not laps:
        raise typer.BadParameter(f"no saved laps for {track.value} (run kmslap laptime {track.value})")
    t = TRACK_NAMES[track.value]()
    for name, d in laps.items():
        print(f"{name:52s} {d.attrs['lap_time']:6.2f} s  {d.attrs.get('penalties', '')}")
    if rerun:
        from kmslap.reporting.race3d import show
        print(f"wrote {show(laps, t, save)}")
        return
    birdseye.lap_gif(laps, track=t, out=out, speedup=speedup, title=f"{t.name}: flying laps")
    print(f"wrote {out}")


@app.command()
def mcap(track: Annotated[TrackName, typer.Argument()] = TrackName.ac18,
         laps: int = 2, grip: float = 0.85,
         margin: Annotated[float, typer.Option(help="m kept between the car's side and the edge / cones")] = 0.35,
         width: Annotated[Optional[float], typer.Option(help="track width, m (michigan2026)")] = None,
         out: str = "sim.mcap",
         start: Annotated[Optional[str], typer.Option(help="log start time, ISO (default now)")] = None):
    """A simulated log in the car logger's MCAP format (same topics / schemas), for Foxglove."""
    from kmslap.reporting.mcap_log import run
    run(track.value, laps, grip, margin, width=width, out=out, start=start)


@app.command()
def birdseye(out: str = "birdseye", gif: Annotated[bool, typer.Option(help="also the lap GIF (~1 min)")] = True):
    """Endurance strategy presets racing a flying lap on Michigan 2026: GIF + speed map."""
    from kmslap.reporting.birdseye import run
    run(out, gif=gif)


@app.command()
def strategy(out: str = "strategy", workers: Workers = None,
             regen: Annotated[bool, typer.Option(help="also search regen strategies")] = False,
             remote: Remote = [],
             soc_floor: Annotated[float, typer.Option(help="SOC reserve the event must end above")] = 0.10):
    """Tune pace, lift-and-coast, DRS (and regen) for the fastest finishing event."""
    from kmslap.strategy import run
    run(out, workers, regen, remote, soc_floor)


@app.command()
def bench(what: Bench, n: Annotated[int, typer.Option(help="strategy evaluations")] = 96,
          workers: Workers = None, remote: Remote = [], out: str = "bench"):
    """Benchmarks: single-thread timings, parallel throughput, or the cluster."""
    from kmslap.ops.bench import record
    record(what.value, n, workers, remote, out)


@app.command()
def dyno(out: str = "dyno"):
    """Dyno pulls vs the inverter logs and the motor model (validation.dyno)."""
    from kmslap.validation.dyno import run_all
    run_all(out)


@app.command()
def aero(out: str = "aero"):
    """2024 aero tests: CdA from coast-downs, ClA and balance from shock pots (validation.aero)."""
    from kmslap.validation.aero import run
    run(out)


@app.command("pack-temps")
def pack_temps(out: str = "pack",
               gif: Annotated[bool, typer.Option(help="also render the top-down heatmap GIFs (~1 min)")] = True):
    """KS6e accumulator thermal tests: top-down heatmaps, lumped model fit (validation.pack_thermal)."""
    from kmslap.validation.pack_thermal import run
    run(out, gif)


# --- remote ---------------------------------------------------------------------------

@remote_app.command("setup")
def remote_setup(host: Host = _remote_defaults.DEFAULT_HOST, dir: Dir = _remote_defaults.DEFAULT_DIR):
    """Check git / nix there, clone, build the environment."""
    _remote_defaults.setup(host, dir)


@remote_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def remote_run(ctx: typer.Context,
               fetch: Annotated[list[str], typer.Option(help="copy this path back afterwards (repeatable)")] = [],
               detach: Annotated[bool, typer.Option(help="start it and return; see `log` / `fetch`")] = False,
               uv: Annotated[bool, typer.Option(help="`uv run` instead of nix develop")] = False,
               host: Host = _remote_defaults.DEFAULT_HOST, dir: Dir = _remote_defaults.DEFAULT_DIR):
    """Run a command there on this working tree: kmslap remote run [opts] -- kmslap strategy ..."""
    _remote_defaults.run(host, dir, list(ctx.args), fetch, detach, uv)


@remote_app.command("fetch")
def remote_fetch(paths: list[str], host: Host = _remote_defaults.DEFAULT_HOST,
                 dir: Dir = _remote_defaults.DEFAULT_DIR):
    """Copy paths back from the remote clone."""
    _remote_defaults.fetch(host, dir, paths)


@remote_app.command("log")
def remote_log(host: Host = _remote_defaults.DEFAULT_HOST, dir: Dir = _remote_defaults.DEFAULT_DIR):
    """Follow a detached job's output."""
    _remote_defaults.log(host, dir)


# --- cluster --------------------------------------------------------------------------

@cluster_app.command("selftest")
def cluster_selftest(n: int = 60, remote: Remote = [], workers: Workers = None):
    """Sleep tasks across this machine and remotes, to try the live dashboard."""
    from kmslap.ops.cluster import selftest
    selftest(n, remote, workers)


@cluster_app.command("node", hidden=True)
def cluster_node(address: Annotated[str, typer.Option()], authkey: Annotated[str, typer.Option()],
                 name: Annotated[str, typer.Option()], workers: Annotated[int, typer.Option()]):
    """(started by kmslap.ops.cluster) run workers against a coordinator."""
    from kmslap.ops.cluster import node
    node(address, bytes.fromhex(authkey), name, workers)


def main():
    app(prog_name="kmslap")


if __name__ == "__main__":
    main()
