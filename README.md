# python-data-analysis

scripts to evaluate performance

## Setup (Nix)

```sh
nix develop            # or `direnv allow` once, then just cd in
jupyter lab            # notebooks
python -m motor        # scripts run from repo root (PYTHONPATH is set to it)
```

Python deps come from nixpkgs (see `flake.nix`), so there is no pip/venv step.
`requirements.txt` is kept for reference on non-Nix machines.

Notebook outputs are stripped on commit by `nbstripout` (configured automatically
by the dev shell; see `.gitattributes`). Your local copies keep their outputs.
