"""Tracks (constant width and coned), racing lines within them, penalty scoring, saved
laps, and the minimum-time solvers on a small oval."""
import numpy as np
import pandas as pd
import pytest

from kmslap.driving.paths import Path
from kmslap.events import track as T


def oval(straight=40.0, radius=12.0):
    return Path.from_segments([("straight", straight), ("arc", radius, 180.0), ("straight", straight),
                               ("arc", radius, 180.0)], step=0.5, closed=True)


def test_ac18_cone_course_builds_from_its_cones():
    t = T.michigan_ac18()
    assert len(t.cones) == 167                      # 179 minus the start chute's 12
    assert 1150 < t.center.length < 1210
    width = t.w_left + t.w_right
    assert 4.4 < np.median(width) < 5.0             # gates ~4.8 m between cone centres
    lo, hi = t.limits()
    assert (hi - lo).min() > 1.0                    # nowhere narrower than the car


def test_racing_line_keeps_inside_per_side_limits():
    t = T.constant_width(oval(), 5.0)
    t = T.Track("lopsided", t.center, t.w_left + 1.0, t.w_right - 0.5)
    line = T.racing_line(t)
    lo, hi = t.limits()
    off = T.lateral_offset(t.center, line)
    idx = T._nearest(t.center, line.x, line.y)
    assert (off >= lo[idx] - 0.05).all() and (off <= hi[idx] + 0.05).all()
    assert line.length < t.center.length            # cuts the corners


def test_penalties_count_cones_and_off_course():
    center = Path(np.linspace(0, 100, 201), np.zeros(201))
    n = len(center.s)
    t = T.Track("straight", center, np.full(n, 2.5), np.full(n, 2.5), cones=np.array([[50.0, 0.3], [70.0, 2.5]]))
    x = np.linspace(5, 95, 400)
    clean = T.penalties(t, x, np.zeros_like(x), np.zeros_like(x))
    assert clean.cones == 1 and clean.off_course == 0       # straight through the first cone
    wide = T.penalties(t, x, np.where((x > 20) & (x < 30), 4.0, -1.0), np.zeros_like(x))
    assert wide.cones == 0 and wide.off_course == 1
    assert wide.seconds == 20.0


def test_saved_laps_round_trip(tmp_path, monkeypatch):
    from kmslap.reporting import runs
    monkeypatch.setattr(runs, "RUNS", tmp_path)
    t = np.linspace(0, 10, 101)
    df = pd.DataFrame({"t": t, "X": t * 5, "Y": 0 * t, "psi": 0 * t, "speed": 5 + 0 * t, "pedal": 0.5 + 0 * t,
                       "brake": 0 * t, "motor_torque": 50 + 0 * t})
    runs.save_lap("oval", "a line", df, 2.0, 8.0, penalties="none")
    laps = runs.load_laps("oval")
    d = laps["a line"]
    assert d.t_lap.iloc[0] == 0 and d.t_lap.iloc[-1] == pytest.approx(6.0)
    assert d.dist.iloc[-1] == pytest.approx(30.0)
    assert (d["mode"] == "drive").all() and d.attrs["lap_time"] == pytest.approx(6.0)


@pytest.fixture(scope="module")
def env():
    from kmslap.car.params import Car
    from kmslap.models.envelope import gg_envelope
    return gg_envelope(Car())


def test_min_time_lines_beat_the_centerline_on_an_oval(env):
    from kmslap.models import mintime, mintime_dt
    from kmslap.models.envelope import speed_profile
    track = T.constant_width(oval(), 6.0)
    center_time = np.sum(np.diff(track.center.s, append=track.center.length)
                         / speed_profile(track.center, env, margin=0.85))
    q = mintime.solve(track, env, grip=0.85, step=2.0)
    assert q.stats["success"] and q.lap_time < center_time
    d = mintime_dt.solve(track, env, grip=0.85, step=2.0, warm=q)
    assert d.stats["success"]
    lo, hi = track.resampled(2.0).limits()
    assert (d.x["n"] >= lo - 1e-3).all() and (d.x["n"] <= hi + 1e-3).all()
    assert abs(d.lap_time - q.lap_time) < 0.15 * q.lap_time   # same car, same track
    assert np.abs(d.beta_ff).max() <= np.radians(10) + 1e-6


def _fake_laps(track):
    c = track.center
    laps = {}
    for k, v in enumerate((10.0, 12.0)):
        t = c.s / v
        laps[f"car {k}"] = pd.DataFrame({"t_lap": t, "dist": c.s, "X": c.x, "Y": c.y, "psi": c.heading, "speed": v + 0 * t,
                                         "beta": 0 * t, "steering_wheel": 0.1 + 0 * t, "mode": "drive"})
    return laps


def test_race_renders_in_rerun(tmp_path):
    from kmslap.reporting import race3d
    track = T.constant_width(oval(), 5.0)
    out = race3d.show(_fake_laps(track), track, save=str(tmp_path / "race.rrd"))
    assert (tmp_path / "race.rrd").stat().st_size > 10_000 and out.endswith("race.rrd")


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="no ffmpeg (it's in the Nix dev shell)")
def test_race_video(tmp_path):
    from kmslap.reporting import birdseye
    track = T.constant_width(oval(), 5.0)
    birdseye.lap_gif(_fake_laps(track), track=track, out=str(tmp_path / "race.mp4"), speedup=20.0, fps=5)
    assert (tmp_path / "race.mp4").stat().st_size > 1000


def test_simulated_log_maps_every_sensor_to_the_latest_dbc(tmp_path):
    """Every mapped sensor message, in the logger's schema for the current DBC, with
    values that read back as the sim's; plus the /sim truth."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    from kmslap.car.params import Car
    from kmslap.driving.drive import drive
    from kmslap.driving.follower import PathFollower
    from kmslap.reporting import mcap_log

    track = T.constant_width(oval(), 6.0)
    track.origin = (-84.2371765, 42.0688368)
    f = PathFollower(track.center, speed=12.0, car=Car(), laps=1)
    res = drive(f, Car(), psi0=float(track.center.heading[0]), t_max=8.0, log_period=0.01)
    counts = mcap_log.write(res.df, track, tmp_path / "sim.mcap", preroll_s=1.0, start_ns=1_749_823_200_000_000_000)
    expected = {m.lower() + "_data" for m in mcap_log.SENSORS} | {"m170_internal_states_data", "m192_command_message_data", "vcu_status_data"}
    assert expected <= set(counts)
    assert {"/sim/tf", "/sim/car", "/sim/track", "/sim/state", "/sim/location"} <= set(counts)
    descriptor = (mcap_log.DIR / "hytech.bin").read_bytes()
    last = {}
    with open(tmp_path / "sim.mcap", "rb") as fh:
        r = make_reader(fh, decoder_factories=[DecoderFactory()])
        s = r.get_summary()
        for ch in s.channels.values():
            if not ch.topic.startswith("/sim"):
                assert s.schemas[ch.schema_id].data == descriptor, ch.topic
        for _, ch, _, m in r.iter_decoded_messages(topics=["m172_torque_and_timer_info_data", "an1_fl_wheelspeed_data",
                                                           "an1_rl_loadcell_data", "vcu_status_data"]):
            last[ch.topic] = m
    assert last["m172_torque_and_timer_info_data"].INV_Torque_Feedback > 0          # cruising, motoring
    v = res.df.speed.iloc[-1]
    assert last["an1_fl_wheelspeed_data"].wheel_rpm == pytest.approx(v / Car().tire_radius * 60 / (2 * np.pi), rel=0.05)
    assert 400 < last["an1_rl_loadcell_data"].force_N < 1500                          # a quarter of the car, roughly
    assert last["vcu_status_data"].vcu_statemachine_state == "READY_TO_DRIVE"
