"""Run a job from this repo on another machine (e.g. the desktop's WSL, over
Tailscale), with exactly the code you have here, and bring the results back.

    python -m accel.remote setup                      # once: clone, build the Nix env
    python -m accel.remote run --fetch strategy -- python -m accel.strategy --regen --out strategy
    python -m accel.remote run --detach --fetch strategy -- python -m accel.strategy --out strategy
    python -m accel.remote log                        # follow a detached job
    python -m accel.remote fetch strategy             # collect its results

What `run` does:
  1. snapshots the working tree as a commit (`git stash create`: tracked files
     including uncommitted changes; untracked files are left out, commit or `git add`
     them first), without touching your branch or stash;
  2. pushes it to the remote clone over SSH (to refs/remote-run/latest);
  3. checks it out there (detached) and runs the command in `nix develop` (or
     `uv run` with --uv), streaming its output;
  4. copies the --fetch paths back into this checkout (tar over SSH).

Host: --host, or $ACCEL_REMOTE_HOST, default mathewos@desktop-17id03o-2. The remote needs
git, Nix with flakes (or uv) and SSH from here; see README "Running jobs on another
machine". Build caches (compiled models, envelopes) live in the remote clone's build/.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOST = os.environ.get("ACCEL_REMOTE_HOST", "mathewos@desktop-17id03o-2")
DEFAULT_DIR = os.environ.get("ACCEL_REMOTE_DIR", "python-data-analysis")   # relative to remote $HOME
REPO_URL = "https://github.com/mathbrook/python-data-analysis.git"
REF = "refs/remote-run/latest"
LOG = ".remote-run.log"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def ssh_command(host: str, remote_cmd: str) -> list[str]:
    return ["ssh", *SSH_OPTS, host, remote_cmd]


def snapshot() -> str:
    """A commit of the working tree (tracked files, uncommitted changes included)."""
    return _git("stash", "create") or _git("rev-parse", "HEAD")


def job_script(dir_: str, sha: str, command: list[str], use_uv: bool) -> str:
    """The shell script `run` executes on the remote."""
    cmd = shlex.join(command)
    runner = f"uv run {cmd}" if use_uv else f"nix develop -c {cmd}"
    return (f"set -e; cd {shlex.quote(dir_)}; git checkout -q --detach {sha}; "
            f"echo \"[remote-run] $(hostname): {sha[:10]}: {cmd}\"; {runner}")


def push(host: str, dir_: str, sha: str):
    # push the snapshot to a ref outside branches, so the remote's checkout never blocks it
    subprocess.run(["git", "-C", str(ROOT), "push", "-q", "--force", f"{host}:{dir_}",
                    f"{sha}:{REF}"], check=True,
                   env={**os.environ, "GIT_SSH_COMMAND": shlex.join(["ssh", *SSH_OPTS])})


def fetch(host: str, dir_: str, paths: list[str]):
    """Copy remote paths (relative to the clone) into this checkout."""
    for p in paths:
        tar = subprocess.Popen(ssh_command(host, f"tar -C {shlex.quote(dir_)} -cf - {shlex.quote(p)}"),
                               stdout=subprocess.PIPE)
        subprocess.run(["tar", "-C", str(ROOT), "-xf", "-"], stdin=tar.stdout, check=True)
        if tar.wait():
            raise SystemExit(f"fetching {p} failed")
        print(f"[remote-run] fetched {p}")


def setup(host: str, dir_: str):
    check = ("command -v git >/dev/null || { echo 'missing: git'; exit 1; }; "
             "command -v nix >/dev/null || command -v uv >/dev/null || "
             "{ echo 'missing: nix (or uv)'; exit 1; }; "
             f"[ -d {shlex.quote(dir_)}/.git ] || git clone -q {REPO_URL} {shlex.quote(dir_)}; "
             f"cd {shlex.quote(dir_)} && "
             "if command -v nix >/dev/null; then nix develop -c python -c 'import accel; print(\"env ok\")'; "
             "else uv sync -q && uv run python -c 'import accel; print(\"env ok\")'; fi; "
             "echo cores: $(nproc)")
    sys.exit(subprocess.run(ssh_command(host, check)).returncode)


def run(host: str, dir_: str, command: list[str], fetch_paths: list[str], detach: bool, use_uv: bool):
    if not command:
        raise SystemExit("nothing to run: python -m accel.remote run -- <command>")
    sha = snapshot()
    untracked = _git("ls-files", "--others", "--exclude-standard")
    if untracked:
        print("[remote-run] not sent (untracked): " + ", ".join(untracked.splitlines()[:8]), file=sys.stderr)
    push(host, dir_, sha)
    script = job_script(dir_, sha, command, use_uv)
    if detach:
        log = f"{dir_}/{LOG}"
        remote = f"nohup sh -c {shlex.quote(script)} > {shlex.quote(log)} 2>&1 < /dev/null &"
        subprocess.run(ssh_command(host, remote), check=True)
        print(f"[remote-run] started on {host}; `python -m accel.remote log` to follow, "
              f"`python -m accel.remote fetch {' '.join(fetch_paths) or 'PATH'}` when done")
        return
    rc = subprocess.run(ssh_command(host, script)).returncode
    if fetch_paths:
        fetch(host, dir_, fetch_paths)
    sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="the remote clone, relative to its $HOME")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("setup", help="check git/nix, clone, build the environment")
    r = sub.add_parser("run", help="run a command there on this working tree")
    r.add_argument("--fetch", action="append", default=[], metavar="PATH",
                   help="copy this path back afterwards (repeatable)")
    r.add_argument("--detach", action="store_true", help="start it and return; see log / fetch")
    r.add_argument("--uv", action="store_true", help="run with `uv run` instead of nix develop")
    r.add_argument("command", nargs=argparse.REMAINDER)
    f = sub.add_parser("fetch", help="copy paths back from the remote clone")
    f.add_argument("paths", nargs="+")
    sub.add_parser("log", help="follow a detached job's output")
    args = ap.parse_args()

    if args.action == "setup":
        setup(args.host, args.dir)
    elif args.action == "run":
        cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
        run(args.host, args.dir, cmd, args.fetch, args.detach, args.uv)
    elif args.action == "fetch":
        fetch(args.host, args.dir, args.paths)
    else:
        sys.exit(subprocess.run(ssh_command(args.host, f"tail -n +1 -f {shlex.quote(args.dir)}/{LOG}")).returncode)


if __name__ == "__main__":
    main()
