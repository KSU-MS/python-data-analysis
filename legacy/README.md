# legacy/

The repo's older notebooks and scripts (from before `kmslap`), kept for reference and
reviewed below for what's worth bringing back. Most read logs from Windows paths
(`C:\Users\Matthew Samson\...`); those logs now live in the team log archive (bottom).
Code here imports the old root modules (`motor`, `battery`, `tire`), which moved to
`kmslap/car/`, so it won't run as-is.

## Review

| Item | What it is | Verdict |
|---|---|---|
| `acc_heatmap/` (`heatmap.py`, `log2heatmap.ipynb`, `manipulate_df.py`, `plot.py`, `plot_slider.py`) + `cellarraycsv.csv` | Accumulator cell temperatures: thermistor calibration (volts → 8-bit ADC counts → 5th-order polynomial), the pack's top-down 6×12 layout, an animated heatmap with a slider | **Revived** as `kmslap pack-temps` (`kmslap/validation/pack_thermal.py`): same calibration (checked against the converted `temps_real.csv` to 0.0 K) and layout, heatmap GIFs, plus a fit of the sim's lumped pack thermal model to the tests |
| `motor_power_graphs.ipynb`, `motor_power_graphs_6ePowerPull.ipynb`, `torque_log_analysis.py` | Dyno pulls vs inverter logs: aligned by peak power, power by several routes (pack, DC, Vd·Id + Vq·Iq, Iq·Kt), torque | **Revived** as `kmslap dyno` (`kmslap/validation/dyno.py`): aligned by speed instead (r = 0.9997), compared with the motor map and efficiency map. The pull windows are in `cardata/dyno/` |
| `efficiency_plot/` | Hand digitization of the EMRAX 228 efficiency map's contours | **Superseded** by `datasheets/emrax_228hv_efficiency.csv`; used as an independent check: on each traced contour the new map's median is within 0.005 of its label |
| `dynoruns/` | `DJDataExport_DYNORUN_015.txt` (the 3-3-24 228HV pull again, raw export) and `228_240pULL_3-3-24.txt` (summary only, no data) | Keep for reference; the usable pulls moved to `cardata/dyno/` |
| `motorandbatterysim.ipynb`, `motor_curves/` | Torque curves of each EMRAX on the Energus pack, with and without voltage sag | **Superseded** by `kmslap/car/powertrain.py` (motor map at the actual bus voltage, pack sag) |
| `batterySimTest.ipynb` | Simulated vs logged pack current / power for a run | Superseded by the double-track model's pack; the idea (replay a logged torque request through the pack model, compare voltage) is a good validation to redo against `kmslap` |
| `compare_cars_torque.ipynb` | Max torque vs rpm of two cars' logs side by side (Id/Iq, commanded torque) | Revive if comparing cars' torque envelopes again; easy on top of `validation/dyno.py`'s log loading |
| `car_run_analysis.ipynb`, `ic_car_run_analysis.ipynb`, `omnilogviewer.ipynb` | General log browsing and filtering for EV / IC / "omni" logger logs | Drop; a log-viewer in `kmslap` would start from the archive's parsed CSVs |
| `accel_weight_transfer.ipynb` | The original accel sim | Superseded by `kmslap/events/accel.py` (kept for its history) |
| `tire_playground.ipynb` | Pacejka fit exploration | Keep as a scratchpad; the fits live in `kmslap/car/pacejka/` |
| `autopid.py` | A PID helper class | Drop (the controllers live in `controller/` and `kmslap/vcu/`) |
| `wiresizecalc.py` | Wire resistance / heating estimate | Keep (small, standalone) |
| `plotserial.py` | Live plot from a serial port | Drop |
| `csv_crawler.py`, `path_of_csvs*.json`, `parser_utils/` | Finding and parsing logs on the old laptop; DBC download from KSU-MS/ksu-ms-dbc | Drop (the parser lives with the data logger) |

## Team log archive

The old laptop's formula folder (63 GB): `/run/media/matty/1TB/old_oxos_laptop_user/formula`,
mostly `MatthewS/ks6e_logs/` and dated folders of parsed CSVs (one column per CAN signal,
each sparse; epoch-ms `Time`). Reviewed for what's still relevant to the current car:

| Folder | For |
|---|---|
| `7_6_aero_test_with_aero/`, `7_6_aero_test_no_aero/` | **Used**: `kmslap aero` (`cardata/aero/`). 6 July 2024 (KS7e), same day with and without aero. Parsed logs have VectorNav (velocity, accel, attitude), all four shock pots and wheel speeds, motor speed, front/rear brake pressure, steering: CdA from coast-downs or power vs speed, ClA and balance from shock deflection vs speed^2 (with spring / motion ratios). Pertinent if the current aero is close to 2024's |

Not pursued: the 2023 accumulator thermal tests (an old accumulator design), the 2023
endurance / competition, regen-day and accel / g-g logs, the other dyno days (KS6e-era
car, and the `.drf` WinPEP files would need exporting), the 2022 aero tests.
