"""The `accel` command: every command parses, choices match the model, hosts resolve."""
import pytest
from typer.testing import CliRunner

from kmslap import cli
from kmslap.ops import remote
from kmslap.car.powertrain import MOTORS, PACKS

runner = CliRunner()
COMMANDS = [[], ["accel"], ["report"], ["report-diff"], ["junit-summary"], ["viz"], ["drive"], ["endurance"], ["strategy"],
            ["bench"], ["dyno"], ["aero"], ["pack-temps"], ["laptime"], ["race"], ["mcap"], ["hil"], ["birdseye"], ["remote"], ["remote", "run"], ["remote", "setup"], ["remote", "fetch"],
            ["remote", "log"], ["cluster"], ["cluster", "selftest"]]


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: " ".join(c) or "accel")
def test_help(cmd):
    r = runner.invoke(cli.app, [*cmd, "--help"])
    assert r.exit_code == 0, r.output
    assert "Usage" in r.output


def test_choices_match_the_model():
    # listed in cli.py so completion doesn't import the model
    assert {m.value for m in cli.Motor} == set(MOTORS)
    assert {p.value for p in cli.PackName} == set(PACKS)
    from kmslap.events.track import TRACK_NAMES
    assert {t.value for t in cli.TrackName} == set(TRACK_NAMES)


def test_accel_runs():
    r = runner.invoke(cli.app, ["accel", "3.0"])
    assert r.exit_code == 0, r.output
    assert "ideal TC" in r.output


def test_remote_run_passes_the_command_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote, "run", lambda host, dir_, cmd, fetch, detach, uv: seen.update(
        host=host, cmd=cmd, fetch=fetch, detach=detach))
    r = runner.invoke(cli.app, ["remote", "run", "--fetch", "out", "--host", "h", "--",
                                "accel", "strategy", "--regen", "--out", "out"])
    assert r.exit_code == 0, r.output
    assert seen == {"host": "h", "cmd": ["accel", "strategy", "--regen", "--out", "out"],
                    "fetch": ["out"], "detach": False}


def test_bare_host_gets_the_default_user():
    user, host = remote.DEFAULT_HOST.split("@")
    assert remote.resolve(host) == f"{user}@{host}"
    assert remote.resolve("someone@else") == "someone@else"
    assert remote.resolve("other-machine") == "other-machine"


def test_host_completion_includes_the_default():
    assert remote.DEFAULT_HOST in cli.complete_hosts("")


def test_completion_protocol_used_by_completions_dir():
    # completions/_kmslap (installed with `nix profile install .#completions`) asks
    # kmslap through these variables; the answer is zsh completion code
    r = runner.invoke(cli.app, [], prog_name="kmslap",
                      env={"_KMSLAP_COMPLETE": "complete_zsh", "_TYPER_COMPLETE_ARGS": "kmslap st"})
    assert "strategy" in r.output
    from pathlib import Path
    text = (Path(cli.__file__).resolve().parent.parent / "completions" / "_kmslap").read_text()
    assert text.startswith("#compdef kmslap") and "_KMSLAP_COMPLETE=complete_zsh" in text
