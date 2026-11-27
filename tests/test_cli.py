"""The `accel` command: every command parses, choices match the model, hosts resolve."""
import pytest
from typer.testing import CliRunner

from accel import cli, remote
from accel.powertrain import MOTORS, PACKS

runner = CliRunner()
COMMANDS = [[], ["sim"], ["report"], ["report-diff"], ["viz"], ["drive"], ["endurance"], ["strategy"],
            ["bench"], ["remote"], ["remote", "run"], ["remote", "setup"], ["remote", "fetch"],
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


def test_sim_runs():
    r = runner.invoke(cli.app, ["sim", "3.0"])
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
