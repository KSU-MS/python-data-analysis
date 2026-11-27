"""Validation against car data: dyno pulls and the accumulator thermal tests."""
import numpy as np
import pytest

from kmslap.validation import dyno, pack_thermal


def test_dyno_ks6e_matches_the_model():
    c = dyno.compare("ks6e_2023-04-26")
    s = c.summary
    assert s["speed_correlation"] > 0.999
    # the fitted roller radius at FD 3.0 is the sim's loaded tire radius
    assert s["effective_tire_radius_m"] == pytest.approx(0.229, abs=0.005)
    # DC -> wheels efficiency: motor map x inverter x driveline within a few percent
    assert s["overall wheels/DC model"] == pytest.approx(s["overall wheels/DC measured"], abs=0.03)
    assert s["inverter torque feedback / dyno torque"] == pytest.approx(1.0, abs=0.1)


def test_dyno_parsers_handle_every_layout():
    for run in dyno.RUNS.values():
        d = dyno.load_dynojet(dyno.DYNO / run.dyno_file)
        assert len(d) > 500 and d.v.max() > 20 and np.nanmax(d.wheel_power) > 1e4


def test_thermistor_calibration_matches_the_converted_log():
    import pandas as pd
    raw = pd.read_csv(pack_thermal.TESTS["no-fans"], usecols=["cell1temp"]).cell1temp
    real = pd.read_csv(str(pack_thermal.TESTS["no-fans"]).replace("temps_raw", "temps_real"),
                       usecols=["cell1temp"]).cell1temp
    m = raw.notna() & real.notna()
    np.testing.assert_allclose(pack_thermal.thermistor_c(raw[m].to_numpy()), real[m], atol=1e-3)


def test_lumped_thermal_model_fits_the_no_fans_test():
    f = pack_thermal.fit("no-fans")
    assert f.rms_k < 1.0
    assert 5 < f.h_a < 25                       # passive pack, W/K
    assert 0.2 < f.resistance < 0.6             # ohm, 72s8p (datasheet-based 0.29)
    assert set(pack_thermal.LAYOUT.ravel()) == set(range(1, 73))
