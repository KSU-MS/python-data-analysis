"""kmslap.ops.cluster on this machine only: ordering, errors, the timeline."""
import pytest

from kmslap.ops.cluster import Cluster


def test_map_in_order_and_timeline(tmp_path):
    with Cluster(local_workers=3, dashboard=False) as c:
        out = c.map("kmslap.ops.cluster:sleep_task", [0.3, 0.1, 0.2, 0.05, 0.15, 0.25])
    assert out == [0.3, 0.1, 0.2, 0.05, 0.15, 0.25]
    assert len(c.history) == 6 and {h[2] for h in c.history} <= {0, 1, 2}
    c.save_timeline(tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 1000


def test_task_errors_surface():
    with Cluster(local_workers=2, dashboard=False) as c:
        with pytest.raises(RuntimeError, match="ValueError"):   # message wording varies by Python
            c.map("math:sqrt", [4.0, -1.0])
