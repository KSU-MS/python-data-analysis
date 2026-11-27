"""Progress for the CLI's long jobs: a line per job on stderr with a spinner, a bar
(when the amount of work is known), what it's on, and the time spent / left.

    with progress.task("rendering race.mp4", total=n) as t:
        for ...: t.advance()
    with progress.task("lap plan (Ipopt)") as t:      # amount unknown: a spinner
        t.note("iteration 120")

Shown only in a terminal and in the main process: tests, logs and pool workers stay
quiet (a no-op task). Nested tasks share one display; prints show above it.
"""
import contextlib
import multiprocessing
import sys

_progress = None
_depth = 0


def enabled() -> bool:
    return sys.stderr.isatty() and multiprocessing.current_process().name == "MainProcess"


class _Task:
    def __init__(self, prog, tid, description):
        self._p, self._id, self._desc = prog, tid, description     # escaped already

    def advance(self, n: float = 1):
        self._p.advance(self._id, n)

    def update(self, completed: float | None = None, total: float | None = None):
        self._p.update(self._id, completed=completed, total=total)

    def note(self, text: str):
        """What it's on, after the description."""
        from rich.markup import escape
        self._p.update(self._id, description=f"{self._desc} [dim]{escape(text)}[/]")


class _Quiet:
    def advance(self, n=1):
        pass

    def update(self, completed=None, total=None):
        pass

    def note(self, text):
        pass


@contextlib.contextmanager
def task(description: str, total: float | None = None):
    """A progress line while the block runs; total: the work (None: unknown)."""
    global _progress, _depth
    if not enabled():
        yield _Quiet()
        return
    from rich.console import Console
    from rich.progress import (BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
    if _progress is None:
        # the times before the description: a long one is cut at the end, not the numbers
        _progress = Progress(SpinnerColumn(), BarColumn(bar_width=24), TaskProgressColumn(),
                             TimeElapsedColumn(), TextColumn("[dim]left[/]"), TimeRemainingColumn(),
                             TextColumn("{task.description}"), console=Console(stderr=True))
        _progress.start()
    _depth += 1
    from rich.markup import escape
    description = escape(description)                 # names may hold [brackets]
    tid = _progress.add_task(description, total=total)
    try:
        yield _Task(_progress, tid, description)
    finally:
        t = _progress.tasks[[x.id for x in _progress.tasks].index(tid)]
        _progress.update(tid, completed=t.total if t.total is not None else 1,
                         total=t.total if t.total is not None else 1)
        _depth -= 1
        if _depth == 0:
            _progress.stop()
            _progress = None
