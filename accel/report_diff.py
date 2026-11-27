"""Compare two accel.report metrics.json files as a markdown table.

    python -m accel.report_diff base/metrics.json head/metrics.json [--artifact-url URL]

CI runs the report on a pull request's base and head and posts this as a PR
comment. The sims are deterministic, so any change is from the code, not noise.
"""
import argparse
import json
from pathlib import Path

MARKER = "<!-- accel-report-diff -->"
REL_TOL = 1e-4  # float formatting noise; anything bigger is a real change


def _fmt(v, unit):
    if v is None:
        return "didn't finish"
    return f"{v:.3f} {unit}".strip() if unit != "% over ideal" else f"{v:+.2f}%"


def _change(b, h, better, unit):
    """(cell text, -1 worse / 0 same / +1 better)"""
    if b is None and h is None:
        return "", 0
    if b is None or h is None:
        return ("now finishes", 1) if b is None else ("no longer finishes", -1)
    if abs(h - b) <= REL_TOL * max(abs(b), 1e-9):
        return "", 0
    if better not in ("lower", "higher"):
        sign = 0   # informational, e.g. a sensitivity
    else:
        sign = 1 if (h < b) == (better == "lower") else -1
    if unit.startswith("%"):
        return f"{h - b:+.2f} pts", sign  # a % of a % reads badly
    rel = f" ({100 * (h / b - 1):+.1f}%)" if b else ""
    return f"{h - b:+.3f}{rel}", sign


def diff(base: dict, head: dict, artifact_url: str | None = None) -> str:
    changed, same = [], []
    for key in head.keys() | base.keys():
        if key not in base or key not in head:
            changed.append((0, key, "new" if key not in base else "removed",
                            _fmt(base[key]["value"], base[key]["unit"]) if key in base else "",
                            _fmt(head[key]["value"], head[key]["unit"]) if key in head else ""))
            continue
        b, h = base[key], head[key]
        text, sign = _change(b["value"], h["value"], h["better"], h["unit"])
        row = (sign, key, text, _fmt(b["value"], b["unit"]), _fmt(h["value"], h["unit"]))
        (changed if text else same).append(row)

    worse = sum(r[0] < 0 for r in changed)
    better = sum(r[0] > 0 for r in changed)
    icon = {1: "🟢", -1: "🔴", 0: "⚪"}
    out = [MARKER, "## Accel sim: this PR vs base", ""]
    if not changed:
        out.append(f"No change in any of the {len(same)} metrics.")
    else:
        out += [f"**{better} better, {worse} worse**, {len(same)} unchanged.", "",
                "| | metric | base | this PR | change |", "|---|---|---|---|---|"]
        for sign, key, text, b, h in sorted(changed, key=lambda r: (r[0], r[1])):
            out.append(f"| {icon[sign]} | {key} | {b} | {h} | {text} |")
    if same:
        out += ["", "<details><summary>Unchanged</summary>", "",
                "| metric | value |", "|---|---|"]
        out += [f"| {key} | {h} |" for _, key, _, _, h in sorted(same, key=lambda r: r[1])]
        out += ["", "</details>"]
    if artifact_url:
        out += ["", f"Drag race GIF and plots: `accel-report` artifact on [this run]({artifact_url})."]
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base")
    p.add_argument("head")
    p.add_argument("--artifact-url")
    args = p.parse_args()
    head = json.loads(Path(args.head).read_text())
    base_path = Path(args.base)
    if not base_path.exists():
        print(f"{MARKER}\n## Accel sim: this PR vs base\n\nThe base branch's report has no "
              "`metrics.json` (it predates the comparison), so there's nothing to compare yet.")
        return
    print(diff(json.loads(base_path.read_text()), head, args.artifact_url), end="")


if __name__ == "__main__":
    main()
