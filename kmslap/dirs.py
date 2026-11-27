"""Where things live (kmslap runs from its checkout; the editable install puts the repo
root on sys.path):

  build/   caches, regenerated as needed: compiled models, grip envelopes, lap plans,
           racing lines, tire tables. Safe to delete.
  out/     what the commands make, one folder per command (out/report, out/race, ...),
           and the saved laps (out/laps/<track>) that kmslap race races.
Both gitignored. A command's --out puts its output elsewhere instead.
"""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
BUILD = REPO / "build"              # caches (gitignored)
OUT = REPO / "out"                  # the commands' outputs (gitignored)
LAPS = OUT / "laps"                 # saved laps, per track
DATASHEETS = REPO / "datasheets"
TRACKS = REPO / "tracks"
CONTROLLER = REPO / "controller"    # the VCU controller's C++


def out(command: str, given: str | None = None, file: str | None = None) -> Path:
    """Where a command writes: `given` (its --out, as typed) if set, else out/<command>
    (a folder), or out/<command>/<file> for a command that writes one file. The
    folder is made."""
    path = Path(given) if given else OUT / command / (file or "")
    (path.parent if file or (given and Path(given).suffix) else path).mkdir(parents=True, exist_ok=True)
    return path
