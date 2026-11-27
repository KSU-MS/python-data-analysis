"""accel.remote's pieces that don't need a remote: snapshot and the job script."""
import shlex
import subprocess

from accel import remote


def test_snapshot_is_a_commit_of_the_tree():
    sha = remote.snapshot()
    assert subprocess.run(["git", "-C", str(remote.ROOT), "cat-file", "-t", sha],
                          capture_output=True, text=True).stdout.strip() == "commit"
    # it doesn't move HEAD or touch the stash list
    assert remote._git("rev-parse", "HEAD") != "" and sha


def test_job_script():
    s = remote.job_script("python-data-analysis", "abc123", ["python", "-m", "accel.strategy",
                                                             "--out", "my dir"], use_uv=False)
    assert "checkout -q --detach abc123" in s
    assert "nix develop -c python -m accel.strategy --out 'my dir'" in s
    assert s.startswith("set -e")
    assert "uv run python" in remote.job_script("d", "abc", ["python"], use_uv=True)
    # the whole script survives being passed as one ssh argument
    assert shlex.split(shlex.quote(s)) == [s]
