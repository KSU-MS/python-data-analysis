"""Where things live: the repo's data and build directories (kmslap runs from its
checkout; the editable install puts the repo root on sys.path)."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
BUILD = REPO / "build"              # generated, gitignored: compiled models, caches
DATASHEETS = REPO / "datasheets"
TRACKS = REPO / "tracks"
CONTROLLER = REPO / "controller"    # the VCU controller's C++
