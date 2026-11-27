"""pytest's JUnit XML as a markdown test results table, for the CI job summary.

    pytest --junitxml=report/tests/junit.xml --html=report/tests/pytest.html --self-contained-html
    kmslap junit-summary report/tests/junit.xml [--artifact-url URL]

Counts, duration, and every failure with the end of its traceback. The full
pytest-html report (pytest.html) goes in the same artifact as the sim report.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

TRACE_LINES = 25   # of each failure's traceback, in the summary; pytest.html has all of it
SLOWEST = 5


def _cases(root):
    """(node id, outcome, seconds, message, text) for every test case."""
    for case in root.iter("testcase"):
        # classname is the dotted module (plus class): tests.test_cli -> test_cli::test_help[report]
        cls, name = case.get("classname", ""), case.get("name", "")
        node = "::".join([cls.removeprefix("tests."), name]) if cls else name
        outcome, message, text = "passed", "", ""
        for tag in ("failure", "error", "skipped"):
            el = case.find(tag)
            if el is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message, text = el.get("message", ""), (el.text or "").strip()
                break
        yield node, outcome, float(case.get("time", 0) or 0), message, text


def summarize(xml_text: str, artifact_url: str | None = None) -> str:
    root = ET.fromstring(xml_text)
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    duration = sum(float(s.get("time", 0) or 0) for s in suites)
    cases = list(_cases(root))
    count = {k: sum(c[1] == k for c in cases) for k in ("passed", "failed", "error", "skipped")}
    bad = count["failed"] + count["error"]

    head = ("✅ All tests passed" if not bad else f"❌ {bad} test{'s' if bad > 1 else ''} failed")
    out = ["## Tests", "",
           f"**{head}**: {count['passed']} passed, {count['failed']} failed, {count['error']} errors, "
           f"{count['skipped']} skipped, in {duration:.1f} s.", "",
           "| result | tests |", "|---|--:|",
           f"| ✅ passed | {count['passed']} |", f"| ❌ failed | {count['failed']} |",
           f"| 💥 error (in setup or teardown) | {count['error']} |", f"| ⏭️ skipped | {count['skipped']} |",
           f"| **total** | **{len(cases)}** |", f"| duration | {duration:.1f} s |"]

    failures = [c for c in cases if c[1] in ("failed", "error")]
    if failures:
        out += ["", "### Failures", ""]
        for node, outcome, secs, message, text in failures:
            first = message.splitlines()[0] if message else outcome
            out += [f"<details><summary><code>{node}</code>: {_html(first[:200])}</summary>", "",
                    "```", *_tail(text), "```", "", "</details>", ""]
        out.pop()

    skipped = [c for c in cases if c[1] == "skipped"]
    if skipped:
        out += ["", "<details><summary>Skipped</summary>", "", "| test | reason |", "|---|---|"]
        out += [f"| `{node}` | {_cell(message)} |" for node, _, _, message, _ in skipped]
        out += ["", "</details>"]

    slow = sorted(cases, key=lambda c: -c[2])[:SLOWEST]
    if slow:
        out += ["", f"<details><summary>Slowest {len(slow)}</summary>", "",
                "| test | duration (s) |", "|---|--:|"]
        out += [f"| `{node}` | {secs:.1f} |" for node, _, secs, _, _ in slow]
        out += ["", "</details>"]

    where = f"on [this run]({artifact_url})" if artifact_url else "on this run"
    out += ["", f"Full test report (pytest-html, with captured output): `tests/pytest.html` in the "
                f"`accel-report` artifact {where}."]
    return "\n".join(out) + "\n"


def _tail(text):
    """The last TRACE_LINES lines of a traceback: the assertion is at the end."""
    lines = text.replace("```", "'''").splitlines()
    return lines if len(lines) <= TRACE_LINES else ["...", *lines[-TRACE_LINES:]]


def _html(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cell(s):
    return _html(s.splitlines()[0] if s else "").replace("|", "\\|")


def run(xml: str, artifact_url: str | None = None):
    """Print the markdown for a JUnit XML file (a note if pytest never wrote one)."""
    path = Path(xml)
    if not path.exists():
        print(f"## Tests\n\n**❌ No test results**: `{xml}` wasn't written (pytest didn't get as far "
              "as running tests; see the job log).")
        return
    print(summarize(path.read_text(), artifact_url), end="")
