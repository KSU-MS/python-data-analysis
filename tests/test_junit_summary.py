"""kmslap.reporting.junit_summary: pytest's JUnit XML -> the CI job summary's test table."""
import subprocess
import sys

from kmslap.reporting.junit_summary import run, summarize

XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="1" tests="5" time="12.5">
  <testcase classname="tests.test_accel" name="test_fast" time="0.2"/>
  <testcase classname="tests.test_accel" name="test_slow[3.0]" time="9.1"/>
  <testcase classname="tests.test_vcu" name="test_broken" time="1.0">
    <failure message="assert 2.31 &lt; 2.3">def test_broken():
&gt;       assert 2.31 &lt; 2.3
E       assert 2.31 &lt; 2.3</failure>
  </testcase>
  <testcase classname="tests.test_vcu" name="test_setup" time="0.1">
    <error message="failed on setup with &quot;OSError: no compiler&quot;">OSError: no compiler</error>
  </testcase>
  <testcase classname="tests.test_vcu" name="test_teensy" time="0">
    <skipped type="pytest.skip" message="no arm-none-eabi-gcc | not on PATH">skipped</skipped>
  </testcase>
</testsuite></testsuites>"""


def test_counts_and_failures():
    md = summarize(XML, "https://example/run/1")
    assert "❌ 2 tests failed" in md
    assert "| ✅ passed | 2 |" in md and "| ❌ failed | 1 |" in md
    assert "| 💥 error (in setup or teardown) | 1 |" in md and "| ⏭️ skipped | 1 |" in md
    assert "| **total** | **5** |" in md and "12.5 s" in md
    # each failure with its message (escaped for the <summary> tag) and traceback
    assert "<code>test_vcu::test_broken</code>: assert 2.31 &lt; 2.3" in md
    assert "E       assert 2.31 < 2.3" in md
    assert "test_vcu::test_setup" in md
    # a | in a skip reason can't break the table
    assert "no arm-none-eabi-gcc \\| not on PATH" in md
    # slowest first
    assert md.index("test_accel::test_slow[3.0]") < md.index("test_accel::test_fast")
    assert "[this run](https://example/run/1)" in md


def test_all_passed():
    xml = ('<testsuite tests="1" time="0.5"><testcase classname="tests.test_a" name="test_a" time="0.5"/>'
           "</testsuite>")
    md = summarize(xml)
    assert "✅ All tests passed" in md and "Failures" not in md and "Skipped" not in md


def test_missing_xml(tmp_path, capsys):
    run(str(tmp_path / "junit.xml"))
    assert "No test results" in capsys.readouterr().out


def test_real_pytest_run(tmp_path):
    # pytest's own --junitxml output, so the parser follows what pytest actually writes
    (tmp_path / "test_x.py").write_text("import pytest\n"
                                        "def test_ok(): pass\n"
                                        "def test_bad(): assert 1 == 2\n"
                                        "@pytest.mark.skip(reason='later')\n"
                                        "def test_later(): pass\n")
    xml = tmp_path / "junit.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path / "test_x.py"),
                    f"--junitxml={xml}"], cwd=tmp_path, capture_output=True, check=False)
    md = summarize(xml.read_text())
    assert "| ✅ passed | 1 |" in md and "| ❌ failed | 1 |" in md and "| ⏭️ skipped | 1 |" in md
    assert "test_x::test_bad" in md and "assert 1 == 2" in md
