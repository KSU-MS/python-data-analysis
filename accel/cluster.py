"""Spread independent sim jobs over this machine and remote ones, with a live view.

    from accel.cluster import Cluster
    with Cluster(remotes=["mathewos@desktop-17id03o-2"]) as c:
        results = c.map("accel.strategy:evaluate", candidates, labels=[...])
    c.save_timeline("timeline.png")

How: a coordinator here serves a task queue (multiprocessing.managers, stdlib) on
127.0.0.1. Each machine runs a node (`python -m accel.cluster node`) with one worker
per core; workers pull tasks as they free up, so faster machines take more. Remote
nodes get this working tree pushed and started through accel.remote, and reach the
queue through an SSH reverse tunnel (no ports opened anywhere). Tasks are
"module:function" names plus a picklable argument, so both sides run the same code.

The dashboard (rich, in a terminal) shows each machine's per-core load, what every
worker is running and for how long, and overall progress with throughput and ETA;
without a terminal it prints a line every few seconds. save_timeline() plots every
task on every worker and each machine's CPU over the run.
"""
import argparse
import importlib
import multiprocessing as mp
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing.managers import BaseManager, DictProxy
from pathlib import Path

from . import remote

ROOT = Path(__file__).resolve().parent.parent


class _Manager(BaseManager):
    pass


for _name in ("tasks", "results"):
    _Manager.register(_name)
_Manager.register("status", proxytype=DictProxy)


def _client(address, authkey) -> _Manager:
    host, port = address.rsplit(":", 1)
    m = _Manager(address=(host, int(port)), authkey=authkey)
    for attempt in range(60):           # the tunnel / server may still be coming up
        try:
            m.connect()
            return m
        except (ConnectionRefusedError, OSError):
            time.sleep(1.0)
    raise ConnectionError(f"no coordinator at {address}")


# --- node side ------------------------------------------------------------------

def _worker(address, authkey, host, idx):
    m = _client(address, authkey)
    tasks, results, status = m.tasks(), m.results(), m.status()
    key = f"{host}/{idx}"
    while True:
        try:
            item = tasks.get(timeout=0.5)
        except queue.Empty:
            if status.get("_done"):
                return
            continue
        tid, fn, arg, label = item
        t0 = time.time()
        status[key] = (tid, label, t0)
        try:
            mod, name = fn.rsplit(":", 1)
            out = getattr(importlib.import_module(mod), name)(arg)
            results.put((tid, True, out, host, idx, t0, time.time()))
        except Exception:
            results.put((tid, False, traceback.format_exc(), host, idx, t0, time.time()))
        status[key] = None


def _cpu_times():
    """Per-core (busy, total) jiffies from /proc/stat."""
    out = []
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                v = [int(x) for x in line.split()[1:]]
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                out.append((sum(v) - idle, sum(v)))
    return out


def node(address: str, authkey: bytes, host: str, workers: int):
    """Run `workers` workers against the coordinator and report this machine's
    per-core load until they're done."""
    procs = [mp.Process(target=_worker, args=(address, authkey, host, i)) for i in range(workers)]
    for p in procs:
        p.start()
    status = _client(address, authkey).status()
    prev = _cpu_times()
    while any(p.is_alive() for p in procs):
        time.sleep(0.5)
        cur = _cpu_times()
        load = [100.0 * (b1 - b0) / max(t1 - t0, 1) for (b0, t0), (b1, t1) in zip(prev, cur)]
        prev = cur
        try:
            status[f"_cpu/{host}"] = (time.time(), load, workers)
        except (EOFError, ConnectionError, BrokenPipeError):
            break
    for p in procs:
        p.join()


# --- coordinator ----------------------------------------------------------------

class Cluster:
    """Workers on this machine (local_workers, default all cores; 0 = none) and on
    each remote (remote.py's host strings; workers = its core count)."""

    def __init__(self, local_workers: int | None = None, remotes=(), remote_dir: str = remote.DEFAULT_DIR,
                 dashboard: bool = True):
        self.local_workers = os.cpu_count() if local_workers is None else local_workers
        self.remotes = list(remotes)
        self.remote_dir = remote_dir
        self.dashboard = dashboard
        self.history = []          # (tid, host, worker, t0, t1, label)
        self.cpu_history = {}      # host -> [(t, mean load %)]
        self._procs = []

    def __enter__(self):
        self.port = _free_port()
        self.authkey = secrets.token_bytes(16)
        self._tasks, self._results, self._status = queue.Queue(), queue.Queue(), {}
        mgr = _Manager(address=("127.0.0.1", self.port), authkey=self.authkey)
        _Manager.register("tasks", callable=lambda: self._tasks)
        _Manager.register("results", callable=lambda: self._results)
        _Manager.register("status", callable=lambda: self._status, proxytype=DictProxy)
        server = mgr.get_server()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        address = f"127.0.0.1:{self.port}"
        args = ["--address", address, "--authkey", self.authkey.hex()]
        self.hosts = {}
        if self.local_workers:
            name = socket.gethostname()
            self.hosts[name] = self.local_workers
            self._procs.append(subprocess.Popen(
                [sys.executable, "-m", "accel.cluster", "node", *args, "--name", name,
                 "--workers", str(self.local_workers)], cwd=ROOT))
        if self.remotes:
            sha = remote.snapshot()
            for host in self.remotes:
                remote.push(host, self.remote_dir, sha)
                cores = int(subprocess.run(remote.ssh_command(host, "nproc"), capture_output=True,
                                           text=True, check=True).stdout)
                name = host.split("@")[-1]
                self.hosts[name] = cores
                script = remote.job_script(self.remote_dir, sha, [
                    "python", "-m", "accel.cluster", "node", *args, "--name", name,
                    "--workers", str(cores)], use_uv=False)
                cmd = ["ssh", *remote.SSH_OPTS, "-o", "ExitOnForwardFailure=yes",
                       "-R", f"{self.port}:127.0.0.1:{self.port}", host, script]
                self._procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL))
        return self

    def __exit__(self, *exc):
        self._status["_done"] = True
        for p in self._procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()

    def map(self, fn: str, items, labels=None, title: str = "") -> list:
        """fn(item) for every item, spread over all workers; results in order."""
        items = list(items)
        labels = labels or [str(i) for i in range(len(items))]
        for tid, (item, label) in enumerate(zip(items, labels)):
            self._tasks.put((tid, fn, item, label))
        out, done = [None] * len(items), 0
        self._t0 = time.time()
        view = _Dashboard(self, len(items), title or fn) if self.dashboard else None
        if view:
            view.start()
        try:
            while done < len(items):
                try:
                    tid, ok, res, host, idx, t0, t1 = self._results.get(timeout=0.25)
                except queue.Empty:
                    self._sample_cpu()
                    if view:
                        view.update(done)
                    if all(p.poll() is not None for p in self._procs):
                        raise RuntimeError("every node exited with tasks left")
                    continue
                if not ok:
                    raise RuntimeError(f"task {labels[tid]} failed on {host}:\n{res}")
                out[tid] = res
                done += 1
                self.history.append((tid, host, idx, t0, t1, labels[tid]))
                self._sample_cpu()
                if view:
                    view.update(done)
        finally:
            if view:
                view.stop(done)
        return out

    def _sample_cpu(self):
        for key, val in list(self._status.items()):
            if key.startswith("_cpu/"):
                t, load, _ = val
                h = self.cpu_history.setdefault(key[5:], [])
                if not h or h[-1][0] != t:
                    h.append((t, sum(load) / len(load)))

    def save_timeline(self, path):
        """Every task on every worker (one row each, grouped by machine), and each
        machine's mean CPU load over the run."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t_start = min(h[3] for h in self.history)
        rows = {}
        for host in self.hosts:
            for i in range(self.hosts[host]):
                rows[(host, i)] = len(rows)
        colors = dict(zip(self.hosts, ["#468ce6", "#e6504a", "#5abe6e", "#e6aa32"]))
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        for tid, host, idx, t0, t1, label in self.history:
            ax.barh(rows[(host, idx)], t1 - t0, left=t0 - t_start, color=colors[host],
                    edgecolor="white", linewidth=0.5)
        ax.set_yticks(list(rows.values()), [f"{h} {i}" for h, i in rows], fontsize=6)
        ax.invert_yaxis()
        counts = {h: sum(1 for x in self.history if x[1] == h) for h in self.hosts}
        ax.set_title("  ".join(f"{h}: {counts[h]} tasks on {n} workers" for h, n in self.hosts.items()))
        for host, hist in self.cpu_history.items():
            t, load = zip(*hist)
            ax2.plot([x - t_start for x in t], load, color=colors.get(host, "k"), label=host)
        ax2.set(xlabel="s", ylabel="CPU %", ylim=(0, 105))
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


class _Dashboard:
    """Live terminal view; a plain progress line every few seconds without a TTY."""

    def __init__(self, cluster, total, title):
        self.c, self.total, self.title = cluster, total, title
        self.tty = sys.stdout.isatty()
        self._last_print = 0.0

    def start(self):
        if self.tty:
            from rich.live import Live
            self.live = Live(self._render(0), refresh_per_second=4, transient=False)
            self.live.start()

    def update(self, done):
        if self.tty:
            self.live.update(self._render(done))
        elif time.time() - self._last_print > 5:
            self._last_print = time.time()
            print(self._line(done), flush=True)

    def stop(self, done):
        if self.tty:
            self.live.update(self._render(done))
            self.live.stop()
        else:
            print(self._line(done), flush=True)

    def _line(self, done):
        el = time.time() - self.c._t0
        per_host = {h: sum(1 for x in self.c.history if x[1] == h) for h in self.c.hosts}
        load = {k[5:]: sum(v[1]) / len(v[1]) for k, v in self.c._status.items() if k.startswith("_cpu/")}
        hosts = ", ".join(f"{h} {per_host[h]} done{f' cpu {load[h]:.0f}%' if h in load else ' (starting)'}"
                          for h in self.c.hosts)
        return f"[cluster] {done}/{self.total} in {el:.0f} s ({done / max(el, 1e-9):.2f}/s) | {hosts}"

    def _render(self, done):
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        el = time.time() - self.c._t0
        rate = done / el if el > 0 else 0
        eta = (self.total - done) / rate if rate > 0 else float("nan")
        head = Table.grid(expand=True)
        head.add_row(Text(self.title, style="bold"),
                     Text(f"{done}/{self.total}  {el:5.0f} s  {rate:5.2f}/s  ETA {eta:5.0f} s",
                          justify="right"))
        parts = [head, ProgressBar(total=self.total, completed=done)]
        now = time.time()
        for host, n in self.c.hosts.items():
            cpu = self.c._status.get(f"_cpu/{host}")
            bars = Text()
            if cpu:
                for load in cpu[1]:
                    level = min(int(load / 12.5), 7)
                    bars.append("▁▂▃▄▅▆▇█"[level], style="green" if load < 60 else "yellow" if load < 90 else "red")
                bars.append(f"  {sum(cpu[1]) / len(cpu[1]):3.0f}% of {len(cpu[1])} cores")
            else:
                bars.append("starting (pushing code, building the env if it changed)...", style="dim")
            done_here = sum(1 for x in self.c.history if x[1] == host)
            workers = Table.grid(padding=(0, 2))
            for i in range(n):
                cur = self.c._status.get(f"{host}/{i}")
                if cur:
                    workers.add_row(f"{i:2d}", Text(str(cur[1])[:60]), f"{now - cur[2]:5.1f} s")
                else:
                    workers.add_row(f"{i:2d}", Text("idle", style="dim"), "")
            parts.append(Panel(Group(bars, workers), title=f"{host}: {done_here} done, {n} workers",
                               title_align="left"))
        return Group(*parts)


def sleep_task(seconds: float) -> float:
    """A trivial task for trying the cluster: sleep, return the time slept."""
    time.sleep(seconds)
    return seconds


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    n = sub.add_parser("node", help="(started by Cluster) run workers against a coordinator")
    n.add_argument("--address", required=True)
    n.add_argument("--authkey", required=True)
    n.add_argument("--name", required=True)
    n.add_argument("--workers", type=int, required=True)
    t = sub.add_parser("selftest", help="sleep tasks across this machine (and --remote ones)")
    t.add_argument("--n", type=int, default=60)
    t.add_argument("--remote", action="append", default=[])
    t.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.action == "node":
        node(args.address, bytes.fromhex(args.authkey), args.name, args.workers)
    else:
        import random
        tasks = [random.uniform(0.5, 3.0) for _ in range(args.n)]
        with Cluster(local_workers=args.workers, remotes=args.remote) as c:
            c.map("accel.cluster:sleep_task", tasks, labels=[f"sleep {x:.1f} s" for x in tasks],
                  title="cluster self-test")


if __name__ == "__main__":
    main()
