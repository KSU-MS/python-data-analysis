"""Simulated car logs in the team logger's format, for Foxglove: an MCAP of protobuf
messages, one topic per CAN message (`<message>_data`), as if the car's sensors and
boards had sent the simulated values on the bus and the logger had recorded them;
plus ground-truth topics (/sim/...) for Foxglove's 3D, map and plot panels.

    kmslap mcap ac18 --laps 2 --out sim.mcap

The car is the current one: its DBC is KSU-MS/ksu-ms-dbc `main` (cardata/mcap/car.dbc,
@ ffa1f7d, Nov 2026), and the protobuf schema is what ksu_daq's generator makes of it
(dbc_to_proto + protoc: cardata/mcap/hytech.proto / .bin). The chain mirrors the
logger (KSU-MS/ksu_daq, a fork of HyTech's data_acq):
  sim state -> each message's signals in engineering units (SENSORS below)
  -> the frame's raw fields: value / scale, rounded and masked to the field's width
     (so saturation and wrap-around come out as on the wire)
  -> decoded by cantools with the DBC, value tables as names
  -> packed into the logger's protobuf message (its pack_protobuf_msg)
  -> written by mcap_protobuf's Writer to `<message name lowercased>_data`.

SENSORS maps every simulated quantity to the DBC message and signals carrying it, with
each message's rate. Messages the sim has nothing behind (board firmware / config,
charger, inverter parameter access and diagnostics, tire / motor / coolant
temperatures) aren't written: NOT_SIMULATED lists them. Where a sensor's calibration
isn't known (raw ADC counts, shock pot travel), the estimate is in the table's notes and
the file's metadata. Boards are stand-ins; a board's own firmware, run in the loop like
controller/, can replace one.

TODO(firmware in the loop): build the boards' firmware for the PC and run it in the
sim loop in place of the stand-ins, reading the simulated sensors and putting its own
CAN frames in the log. The VCU first (KSU-MS/VCU, `kms-vcu`: raw pedal ADC -> travel ->
plausibility -> state machine -> torque command, so vcu_pedals_travel / vcu_status / M192
become its output); waits on the ota-sil work there. Then the ACU (shutdown / IMD) and
the BMS; the dash and module boards only pack readings, the stand-ins do as well.

Rates: what the June 2025 logs measured for the same message (cardata/mcap/
log_profile.json, under their old names), else an estimate (the DBC has no cycle times).
"""
import json
import math
import time

import numpy as np
import pandas as pd

from kmslap import dirs
from kmslap.car.params import G, Car

DIR = dirs.REPO / "cardata" / "mcap"
POLE_PAIRS = 10                  # EMRAX 228 (the logs: 717 Hz electrical at 4305 rpm)
IQ_PER_NM = 2.55                 # A of Iq per N m (the logs: 578 A at 225.5 N m)
OUTPUT_V_PER_RPM = 0.0371        # inverter output voltage vs speed (the logs)
CELLS = 144                      # in series (Pack.series for the COSMX pack)

# message -> (rate Hz, where the rate is from, the simulated quantities it carries)
SENSORS = {
    "vectornav_attitude":       (100, "logged", "heading (yaw), pitch, roll; VN-300 aligned with the car: X forward, Y right, Z down"),
    "vectornav_gyro":           (100, "logged", "body rates (yaw rate)"),
    "vectornav_acceleration":   (100, "logged", "specific force: longitudinal / lateral g, gravity"),
    "vectornav_velocity":       (100, "logged", "NED velocity"),
    "vectornav_position":       (20, "logged", "lat / lon (the track is georeferenced)"),
    "vectornav_time":           (20, "logged", "unix time"),
    "vectornav_state":          (20, "estimate", "INS state: tracking (2)"),
    "an1_steeringpot":          (100, "estimate", "steering wheel angle; ESTIMATE raw: 4095 counts over 360 deg, 180 deg straight"),
    "an1_front_brakepressure":  (100, "estimate", "front brake torque as pressure; ESTIMATE 7 MPa at full pedal, raw 4095 over 13.8 MPa"),
    "an1_rear_brakepressure":   (100, "estimate", "rear brake pressure, same estimate"),
    "an1_fl_wheelspeed":        (100, "estimate", "wheel speed (rpm)"),
    "an1_fr_wheelspeed":        (100, "estimate", "wheel speed (rpm)"),
    "an1_rl_wheelspeed":        (100, "estimate", "wheel speed (rpm)"),
    "an1_rr_wheelspeed":        (100, "estimate", "wheel speed (rpm)"),
    "an1_fl_loadcell":          (100, "estimate", "tire normal load (N); ESTIMATE raw 4095 over 5 kN"),
    "an1_fr_loadcell":          (100, "estimate", "tire normal load (N)"),
    "an1_rl_loadcell":          (100, "estimate", "tire normal load (N)"),
    "an1_rr_loadcell":          (100, "estimate", "tire normal load (N)"),
    "an1_fl_shockpot":          (100, "estimate", "damper length from the load change; ESTIMATE 40 mm static, 35 N/mm at the damper, raw 4095 over 75 mm"),
    "an1_fr_shockpot":          (100, "estimate", "damper length"),
    "an1_rl_shockpot":          (100, "estimate", "damper length"),
    "an1_rr_shockpot":          (100, "estimate", "damper length"),
    "vcu_pedals_travel":        (20, "logged", "throttle and brake pedal travel (%), steering wheel angle"),
    "vcu_pedal_readings":       (20, "logged", "pedal raw ADC (the June 2025 logs' ranges)"),
    "vcu_status":               (10, "estimate", "state machine (READY_TO_DRIVE while driving), brake active, OK lines"),
    "vcu_distance_tracker_motor": (1, "estimate", "motor energy (Wh), distance, efficiency"),
    "vcu_lifetime_distance_and_ontime": (1, "estimate", "on time, distance"),
    "M165_Motor_Position_Info": (4.1, "logged", "motor speed, electrical angle and frequency"),
    "M166_Current_Info":        (4.1, "logged", "phase currents (from Iq), DC bus current"),
    "M167_Voltage_Info":        (4.1, "logged", "DC bus voltage, output voltage"),
    "M168_Flux_ID_IQ_Info":     (4.1, "logged", "Iq from torque (2.55 A / N m, as logged); Id 0"),
    "M170_Internal_States":     (4.1, "logged", "VSM / inverter state: running, enabled"),
    "M172_Torque_And_Timer_Info": (4.1, "logged", "torque commanded and delivered, on timer"),
    "M173_Modulation_And_Flux_Info": (4.1, "logged", "modulation index, Iq command"),
    "M192_Command_Message":     (10, "logged", "the VCU's torque command, enable, direction"),
    "MSGID_0X6B1":              (125, "logged", "pack current, voltage (loaded, open circuit)"),
    "MSGID_0X6B2":              (125, "logged", "pack temperatures, cell voltages"),
    "MSGID_0X6B3":              (9.6, "logged", "state of charge"),
    "module1_temps":            (0.2, "logged", "cell thermistor volts around the pack temperature; ESTIMATE 10k NTC (B 3435), 15k pull-up from 3.3 V"),
    "module2_temps":            (0.2, "logged", "same"),
    "module3_temps":            (0.2, "logged", "same"),
    "module4_temps":            (0.2, "logged", "same"),
    "module5_temps":            (0.2, "logged", "same"),
    "precharge_status":         (1, "logged", "precharge done"),
}
NOT_SIMULATED = ["acu_shutdown_status", "acu_board_voltage_readings", "vcu_board_readings_one", "vcu_board_readings_two",
                 "vcu_set_parameter", "vcu_pedal_threshold_settings", "dash_buttons", "an1_*_tiretemp", "an1_motor_temp",
                 "an1_coolant_temp", "module*_status", "*_board_data", "MSGID_0X6B4", "charger (MSGID_0X1806E*F4)",
                 "M160-M164, M169, M171, M174-M177 (inverter temperatures, I/O, faults, firmware, diagnostics)",
                 "M193 / M194 (parameter access)", "BMS_Current_Limit", "Status"]


# hub motor inverters (car.hub_front / hub_rear), per wheel: inv_<wheel>_<message> in the
# vehicle's DBC (ksu-ms-dbc gen_tool/vehicles.json; cardata/mcap/vehicles/<drivetrain>/)
HUB_SENSORS = {
    "dti": {
        "erpm_duty_voltage": (100, "estimate", "electrical rpm (4 pole pairs), duty from speed (ESTIMATE), DC input voltage"),
        "ac_dc_current":     (100, "estimate", "AC current peak (sqrt 2 x torque / kt, 0.492 N m/A rms), DC current from the motor's power"),
        "foc":               (100, "estimate", "Iq as the AC current peak, Id 0"),
        "temperatures":      (10, "estimate", "NOT SIMULATED: controller 35 C, motor 45 C held; fault code 0"),
        "misc":              (10, "estimate", "drive enable, no limits active, CAN map version 2.5"),
        "setaccurrent":      (100, "estimate", "the VCU's command: AC current peak for its torque (+ drive, - regen)"),
        "setdriveenable":    (10, "estimate", "the VCU's drive enable"),
    },
    "amk": {
        "actual_values_1":   (200, "estimate", "status (ready, HV and controller enabled), speed, torque current raw (ESTIMATE: in proportion to torque, 16384 at the motor's peak torque)"),
        "actual_values_2":   (200, "estimate", "NOT SIMULATED: motor 45 C, cold plate 35 C, IGBT 40 C held; diagnostic 0"),
        "setpoints_1":       (100, "estimate", "the VCU's command (speed mode with torque limits): 45 m/s setpoint with the positive limit, or 0 rpm with the negative limit for regen, in 0.1 % MN (9.8 N m)"),
    },
}
CASCADIA = ["M165_Motor_Position_Info", "M166_Current_Info", "M167_Voltage_Info", "M168_Flux_ID_IQ_Info",
            "M170_Internal_States", "M172_Torque_And_Timer_Info", "M173_Modulation_And_Flux_Info",
            "M192_Command_Message"]


def hub_wheels(car: Car) -> list:
    return [w for w in ("fl", "fr", "rl", "rr") if (car.hub_front if w[0] == "f" else car.hub_rear)]


def sensors(car: Car) -> dict:
    """SENSORS for this car: without the EMRAX's Cascadia on a car without a central
    motor, with each hub motor's inverter messages."""
    out = {k: v for k, v in SENSORS.items() if car.central_drive or k not in CASCADIA}
    for w in hub_wheels(car):
        for msg, v in HUB_SENSORS[car.hub_motor].items():
            out[f"inv_{w}_{msg}"] = v
    if hub_wheels(car):
        out["vcu_motor_torque_commands"] = (100, "estimate", "the VCU's torque command for each hub motor")
    return out


def vehicle_dir(car: Car):
    """The car's DBC and logger schema: the current car's (main), or its drivetrain
    variant's (cardata/mcap/vehicles/, built from ksu-ms-dbc's vehicle DBCs)."""
    from kmslap.car.drivetrain import name_of
    name = name_of(car)
    return DIR if name == "rwd_emrax" else DIR / "vehicles" / name


def _hub_signals(d: pd.DataFrame, car: Car) -> dict:
    """Each hub motor's inverter messages (DTI HV500 or AMK KW26), from the motor's
    torque, speed and power and the VCU's command."""
    from kmslap.car.drivetrain import HubMotor, motor_sheet
    hub = HubMotor(car.hub_motor)
    sheet = motor_sheet(car.hub_motor)
    n = len(d)
    running = (d.t >= 0).to_numpy()
    bus_v = d.bus_v.to_numpy()
    sig = {}
    cmds = {}
    for w in hub_wheels(car):
        torque = d[f"hub_torque_{w}"].to_numpy()
        cmd = d[f"hub_cmd_{w}"].to_numpy() if f"hub_cmd_{w}" in d else torque
        cmds[w] = cmd
        omega = d[f"omega_{w}"].to_numpy() * car.hub_ratio            # motor rad/s
        rpm = omega * 60 / (2 * np.pi)
        eff = hub.efficiency(rpm, torque)
        p_elec = np.where(torque >= 0, torque * omega / eff, torque * omega * eff)
        dc = p_elec / np.maximum(bus_v, 1.0)
        if car.hub_motor == "dti":
            kt, pp = sheet["torque_constant_nm_per_arms"], sheet["pole_pairs"]
            i_pk = np.sqrt(2) * torque / kt
            sig[f"inv_{w}_erpm_duty_voltage"] = dict(Actual_ERPM=rpm * pp, Actual_InputVoltage=bus_v,
                                                     Actual_Duty=np.clip(100 * rpm / hub.max_rpm, -100, 100))
            sig[f"inv_{w}_ac_dc_current"] = dict(Actual_ACCurrent=i_pk, Actual_DCCurrent=dc)
            sig[f"inv_{w}_foc"] = dict(Actual_FOC_iq=i_pk, Actual_FOC_id=np.zeros(n))
            sig[f"inv_{w}_temperatures"] = dict(Actual_TempController=np.full(n, 35.0), Actual_TempMotor=np.full(n, 45.0),
                                                Actual_FaultCode=np.zeros(n))
            sig[f"inv_{w}_misc"] = dict(Drive_enable=running.astype(float), CAN_map_version=np.full(n, 2.5))
            sig[f"inv_{w}_setaccurrent"] = dict(CMD_TargetAcCurrent=np.where(running, np.sqrt(2) * cmd / kt, 0.0))
            sig[f"inv_{w}_setdriveenable"] = dict(CMD_DriveEnable=running.astype(float))
        else:
            mn = sheet["nominal_torque_nm"]
            # ESTIMATE: torque current in proportion to torque, the converter's peak at the motor's peak
            raw_iq = torque / hub.peak_torque * 16384
            speed_set = 45.0 / car.tire_radius * car.hub_ratio * 60 / (2 * np.pi)
            sig[f"inv_{w}_actual_values_1"] = dict(
                system_ready=np.ones(n), error=np.zeros(n), warning=np.zeros(n), quit_dc_on=running, dc_on=running,
                quit_inverter_on=running, inverter_on=running, derating=np.zeros(n), actual_velocity=rpm,
                torque_current=np.clip(raw_iq, -32768, 32767), magnetizing_current=np.zeros(n))
            sig[f"inv_{w}_actual_values_2"] = dict(temp_motor=np.full(n, 45.0), temp_inverter=np.full(n, 35.0),
                                                   error_info=np.zeros(n), temp_igbt=np.full(n, 40.0))
            sig[f"inv_{w}_setpoints_1"] = dict(
                inverter_on=running, dc_on=running, enable=running, error_reset=np.zeros(n),
                target_velocity=np.where(running & (cmd > 0), speed_set, 0.0),
                torque_limit_positive=np.where(running, np.maximum(cmd, 0) / mn * 100, 0.0),
                torque_limit_negative=np.where(running, np.minimum(cmd, 0) / mn * 100, 0.0))
    if cmds:
        sig["vcu_motor_torque_commands"] = {f"vcu_motor_torque_{w}": cmds.get(w, np.zeros(n))
                                            for w in ("fl", "fr", "rl", "rr")}
    return sig


def _profile():
    return json.loads((DIR / "log_profile.json").read_text())


def _protos(folder=None):
    """{message name: class} from the logs' descriptor set (the current car's, or a
    vehicle folder's)."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    fds = descriptor_pb2.FileDescriptorSet.FromString(((folder or DIR) / "hytech.bin").read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for f in fds.file:
        pool.Add(f)
    out = {}
    for f in fds.file:
        for m in f.message_type:
            out[m.name] = message_factory.GetMessageClass(pool.FindMessageTypeByName(m.name))
    return out


def _wire(msg, values: dict) -> dict:
    """Engineering values -> the frame's raw fields (masked to their widths) -> decoded,
    as the logger sees them."""
    raw = {}
    for sig in msg.signals:
        v = values.get(sig.name, 0)
        if sig.choices and isinstance(v, str):
            names = {str(n): k for k, n in sig.choices.items()}
            r = names[v]
        elif sig.is_float:
            raw[sig.name] = float(v)
            continue
        else:
            r = int(round((float(v) - sig.offset) / sig.scale))
        r &= (1 << sig.length) - 1
        if sig.is_signed and r >> (sig.length - 1):
            r -= 1 << sig.length
        raw[sig.name] = r
    return msg.decode(msg.encode(raw, scaling=False, strict=False), decode_choices=True)


def _pack(cls, decoded: dict):
    """ksu_daq's pack_protobuf_msg: set each field, converting what doesn't fit."""
    pb = cls()
    for key, value in decoded.items():
        try:
            setattr(pb, key, value)
        except TypeError:
            setattr(pb, key, type(getattr(pb, key))(value))
    return pb


def _choice(msg, signal: str, contains: str) -> str:
    """The value-table name of a signal containing `contains` (case-insensitive)."""
    sig = msg.get_signal_by_name(signal)
    return next(str(n) for n in sig.choices.values() if contains.lower() in str(n).lower())


def _ntc_volts(temp_c):
    r = 10e3 * np.exp(3435 * (1 / (np.asarray(temp_c) + 273.15) - 1 / 298.15))
    return 3.3 * r / (r + 15e3)


def _quat(axis, angle):
    angle = np.asarray(angle, float)
    s, c = np.sin(angle / 2), np.cos(angle / 2)
    q = np.zeros(angle.shape + (4,))
    q[..., "xyz".index(axis)] = s
    q[..., 3] = c
    return q


def _vectornav(d: pd.DataFrame, track, rng, noise: dict):
    """VN-300 outputs from the car state, mounted aligned with the car (its body frame
    X forward, Y right, Z down): attitude (deg), body rates (rad/s), specific force
    (m/s^2), NED velocity (m/s), position (deg). Noise: what the parked logs show."""
    psi = d.psi.to_numpy()
    n = len(d)
    yaw_ned = np.degrees(np.pi / 2 - psi)                  # heading from north, clockwise
    c, s = np.cos(psi), np.sin(psi)
    v_e, v_n = c * d.vx - s * d.vy, s * d.vx + c * d.vy
    lon, lat = track.lonlat(d.X.to_numpy(), d.Y.to_numpy()) if track.origin else (0 * psi, 0 * psi)
    z = lambda k: rng.normal(0, noise.get(f"evelogger_vectornav_{k}", 0.0), n)
    return {
        "vectornav_attitude": dict(vn_yaw=(yaw_ned + 180) % 360 - 180 + z("yaw"), vn_pitch=z("pitch"), vn_roll=z("roll")),
        "vectornav_gyro": dict(vn_W_x=z("W_x"), vn_W_y=z("W_y"), vn_W_z=-d.r.to_numpy() + z("W_z")),
        "vectornav_acceleration": dict(vn_accel_x=d.ax.to_numpy() + z("accelX"), vn_accel_y=-d.ay.to_numpy() + z("accelY"),
                                       vn_accel_z=np.full(n, -G) + z("accelZ")),
        "vectornav_velocity": dict(vn_v_n=v_n.to_numpy(), vn_v_e=v_e.to_numpy(), vn_v_d=np.zeros(n)),
        "vectornav_position": dict(vn_latitude=lat, vn_longitude=lon),
        "vectornav_state": dict(inertial_navigation_state=np.full(n, 2)),
    }


def sensor_signals(d: pd.DataFrame, track, car: Car, t0_ns: int, seed: int = 0) -> dict:
    """{DBC message: {signal: array over d.t}}: every simulated quantity on the message
    the current car carries it in (SENSORS)."""
    prof = _profile()
    rng = np.random.default_rng(seed)
    n = len(d)
    t = d.t.to_numpy()
    rate = lambda x: np.gradient(np.asarray(x, float), t)
    running = t >= 0
    rpm = d.motor_rpm.to_numpy()
    torque = d.motor_torque.to_numpy()
    cmd = d.torque_cmd.to_numpy() if "torque_cmd" in d else torque
    theta = np.cumsum(rpm / 60 * POLE_PAIRS * 360 * np.gradient(t)) % 360
    iq = IQ_PER_NM * torque
    out_v = OUTPUT_V_PER_RPM * np.abs(rpm)
    bus_v, bus_i = d.bus_v.to_numpy(), d.bus_i.to_numpy()
    v_oc = d.pack_v_oc.to_numpy() if "pack_v_oc" in d else bus_v
    temp = d.pack_temp.to_numpy()
    dist = np.r_[0, np.cumsum(np.hypot(np.diff(d.X), np.diff(d.Y)))]
    energy_wh = np.cumsum(bus_v * bus_i * np.gradient(t)) / 3600
    steer_deg = np.degrees(d.steering_wheel.to_numpy())
    brake_kpa = 7000.0 * d.brake.to_numpy()
    ped = prof["pedal_raw"]
    raw = lambda k, x: ped[k]["p1"] + np.clip(x, 0, 1) * (ped[k]["p99"] - ped[k]["p1"])
    counts = lambda x, full: np.clip(np.asarray(x) / full, 0, 1) * 4095
    sig = {
        "M165_Motor_Position_Info": dict(INV_Motor_Angle_Electrical=theta, INV_Motor_Speed=rpm,
                                         INV_Electrical_Output_Frequency=rpm / 60 * POLE_PAIRS),
        "M166_Current_Info": dict(INV_Phase_A_Current=np.abs(iq) * np.sin(np.radians(theta)),
                                  INV_Phase_B_Current=np.abs(iq) * np.sin(np.radians(theta - 120)),
                                  INV_Phase_C_Current=np.abs(iq) * np.sin(np.radians(theta + 120)), INV_DC_Bus_Current=bus_i),
        "M167_Voltage_Info": dict(INV_DC_Bus_Voltage=bus_v, INV_Output_Voltage=out_v),
        "M168_Flux_ID_IQ_Info": dict(INV_Id=np.zeros(n), INV_Iq=iq),
        "M172_Torque_And_Timer_Info": dict(INV_Commanded_Torque=cmd, INV_Torque_Feedback=torque,
                                           INV_Power_On_Timer=t - t[0]),
        "M173_Modulation_And_Flux_Info": dict(INV_Modulation_Index=1.53 * out_v / np.maximum(bus_v, 1.0),
                                              INV_Iq_Command=IQ_PER_NM * cmd),
        "MSGID_0X6B1": dict(Pack_Current=bus_i, Pack_Inst_Voltage=bus_v, Pack_Open_Voltage=v_oc, Pack_Summed_Voltage=bus_v),
        "MSGID_0X6B2": dict(High_Temperature=temp + 2, Low_Temperature=temp - 1, Average_Temperature=temp,
                            Avg_Cell_Voltage=v_oc / CELLS, Low_Cell_Voltage=v_oc / CELLS - 0.01),
        "MSGID_0X6B3": dict(Pack_SOC=d.soc.to_numpy() * 100),
        "vcu_pedals_travel": dict(vcu_apps1_travel=d.pedal.to_numpy() * 100, vcu_apps2_travel=d.pedal.to_numpy() * 100,
                                  vcu_bse_travel=d.brake.to_numpy() * 100, steering_angle=steer_deg),
        "vcu_pedal_readings": dict(apps_1_raw=raw("APPS1", d.pedal), apps_2_raw=raw("APPS2", d.pedal), bse_raw=raw("BSE1", d.brake)),
        "vcu_distance_tracker_motor": dict(vcu_motor_energy_wh=energy_wh, vcu_motor_distance_meters=dist,
                                           vcu_motor_efficiency_kmkwh=np.where(energy_wh > 1, dist / np.maximum(energy_wh, 1e-9), 0.0)),
        "vcu_lifetime_distance_and_ontime": dict(vcu_lifetime_ontime=t - t[0], vcu_lifetime_distance=dist),
        "an1_steeringpot": dict(an1_uint12=counts(180 + steer_deg, 360), angle=180 + steer_deg, angle_delta=rate(steer_deg)),
        "an1_front_brakepressure": dict(an1_uint12=counts(brake_kpa, 13800), pressure_kPa=brake_kpa, pressure_delta=rate(brake_kpa)),
        "precharge_status": dict(precharge_state=np.full(n, 3)),
    }
    rear_kpa = brake_kpa * (1 - car.brake_bias_front) / car.brake_bias_front
    sig["an1_rear_brakepressure"] = dict(an1_uint12=counts(rear_kpa, 13800), pressure_kPa=rear_kpa, pressure_delta=rate(rear_kpa))
    for w in ("fl", "fr", "rl", "rr"):
        wheel_rpm = np.abs(d[f"omega_{w}"].to_numpy()) * 60 / (2 * np.pi)
        fz = d[f"fz_{w}"].to_numpy()
        length = 40.0 - (fz - fz[0]) / 35.0                 # compresses under load
        sig[f"an1_{w}_wheelspeed"] = dict(wheel_rpm=wheel_rpm, wheel_rpm_delta=rate(wheel_rpm))
        sig[f"an1_{w}_loadcell"] = dict(an1_uint12=counts(fz, 5000), force_N=fz, force_delta=rate(fz))
        sig[f"an1_{w}_shockpot"] = dict(an1_uint12=counts(length, 75), length_mm=length, length_delta=rate(length))
    for k in range(1, 6):
        spread = rng.normal(0, 0.7, 8)
        sig[f"module{k}_temps"] = {f"module_voltage_{i + 1}": _ntc_volts(temp + spread[i]) for i in range(8)}
    if not car.central_drive:
        for k in CASCADIA:
            sig.pop(k, None)
    if hub_wheels(car):
        sig.update(_hub_signals(d, car))
    sig.update(_vectornav(d, track, rng, prof["vectornav_noise_std"]))
    sig["vectornav_time"] = {"unix_time_ns": t0_ns + ((t - t[0]) * 1e9).astype(np.int64)}
    return sig


def _states(db, d, car: Car = None):
    """State messages: idle (inverter off) before t = 0, running after."""
    vcu = db.get_message_by_name("vcu_status")
    if car is not None and not car.central_drive:            # no Cascadia
        running = (d.t >= 0).to_numpy()
        n = len(d)
        return {"vcu_status": dict(
            vcu_statemachine_state=np.where(running, _choice(vcu, "vcu_statemachine_state", "ready_to_drive"),
                                            _choice(vcu, "vcu_statemachine_state", "tractive_system_energized")).astype(object),
            vcu_brake_active=d.brake.to_numpy() > 0.05, vcu_imd_ok_high=np.ones(n, bool), vcu_bms_ok_high=np.ones(n, bool),
            vcu_bspd_ok_high=np.ones(n, bool), vcu_max_torque=np.full(n, 230))}
    m170, m192 = (db.get_message_by_name(n) for n in ("M170_Internal_States", "M192_Command_Message"))
    running = (d.t >= 0).to_numpy()
    n = len(d)
    pick = lambda msg, sig, on, off: np.where(running, _choice(msg, sig, on), _choice(msg, sig, off)).astype(object)
    cmd = d.torque_cmd.to_numpy() if "torque_cmd" in d else d.motor_torque.to_numpy()
    return {
        "M170_Internal_States": dict(INV_VSM_State=pick(m170, "INV_VSM_State", "motor running", "vsm wait"),
                                     INV_Inverter_State=pick(m170, "INV_Inverter_State", "closed loop", "idle stop"),
                                     INV_Inverter_Enable_State=running, INV_Direction_Command=running,
                                     INV_Inverter_Discharge_State=np.full(n, "Disabled", dtype=object), INV_PWM_Frequency=np.full(n, 12)),
        "M192_Command_Message": dict(VCU_INV_Direction_Command=np.full(n, "CW", dtype=object),
                                     VCU_INV_Inverter_Enable=pick(m192, "VCU_INV_Inverter_Enable", "turn the inverter on", "off"),
                                     VCU_INV_Inverter_Discharge=np.full(n, _choice(m192, "VCU_INV_Inverter_Discharge", "disable"), dtype=object),
                                     VCU_INV_Torque_Command=np.where(running, cmd, 0.0),
                                     VCU_INV_Torque_Limit_Command=np.full(n, 230.0), VCU_INV_Rolling_Counter=np.arange(n) % 16),
        "vcu_status": dict(vcu_statemachine_state=pick(vcu, "vcu_statemachine_state", "ready_to_drive", "tractive_system_energized"),
                           vcu_brake_active=d.brake.to_numpy() > 0.05, vcu_imd_ok_high=np.ones(n, bool),
                           vcu_bms_ok_high=np.ones(n, bool), vcu_bspd_ok_high=np.ones(n, bool), vcu_max_torque=np.full(n, 230)),
    }


def track_scene(track, car=None, plan_line=None):
    """Foxglove SceneUpdate of the track (edges, cones, the planned line), frame 'map'."""
    from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
    from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive

    up = SceneUpdate()
    e = up.entities.add(id="track", frame_id="map")
    lx, ly, rx, ry = track.edges()
    for ex, ey in ((lx, ly), (rx, ry)):
        ln = e.lines.add(type=LinePrimitive.LINE_LOOP, thickness=0.08)
        ln.color.r, ln.color.g, ln.color.b, ln.color.a = 0.9, 0.9, 0.9, 1.0
        for x, y in zip(ex[::2], ey[::2]):
            ln.points.add(x=float(x), y=float(y), z=0.0)
    if plan_line is not None:
        ln = e.lines.add(type=LinePrimitive.LINE_LOOP, thickness=0.05)
        ln.color.r, ln.color.g, ln.color.b, ln.color.a = 0.2, 0.7, 1.0, 0.8
        for x, y in zip(plan_line.x[::2], plan_line.y[::2]):
            ln.points.add(x=float(x), y=float(y), z=0.01)
    if track.cones is not None:
        for x, y in track.cones:
            cy = e.cylinders.add(bottom_scale=1.0, top_scale=0.15)
            cy.pose.position.x, cy.pose.position.y, cy.pose.position.z = float(x), float(y), 0.35
            cy.pose.orientation.w = 1.0
            cy.size.x, cy.size.y, cy.size.z = 0.3, 0.3, 0.7
            cy.color.r, cy.color.g, cy.color.b, cy.color.a = 0.95, 0.55, 0.15, 1.0
    return up




def car_scene(car, steer, ns: int = 0):
    """Foxglove SceneUpdate of the car in its own frame (x forward, z up, origin under
    the CG), front wheels at `steer` (rad): the FSAE car from primitives (car_model),
    team colours. Frame-locked and stamped `ns`: Foxglove places an entity by its frame
    at its timestamp, so without them the car stays where the car frame first was."""
    from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
    from kmslap.reporting import car_model as cm

    up = SceneUpdate()
    e = up.entities.add(id="car", frame_id="car", frame_locked=True)
    e.timestamp.FromNanoseconds(ns)

    def add(part, center=None, q=None):
        c = {"cube": e.cubes, "cylinder": e.cylinders, "sphere": e.spheres}[part.kind].add()
        if part.kind == "cylinder":
            c.bottom_scale, c.top_scale = 1.0, part.top
        c.pose.position.x, c.pose.position.y, c.pose.position.z = center or part.center
        c.pose.orientation.x, c.pose.orientation.y, c.pose.orientation.z, c.pose.orientation.w = (
            float(v) for v in (part.quat if q is None else q))
        c.size.x, c.size.y, c.size.z = part.size
        c.color.r, c.color.g, c.color.b, c.color.a = *part.color, 1.0

    for part in cm.body(car):
        add(part)
    for w in cm.wheels(car):
        turn = cm.quat("z", steer if w.steered else 0.0)
        for part in cm.wheel_parts(car):
            add(part, (w.x, w.y, car.tire_radius), cm.quat_mul(turn, part.quat))
    return up


SIM_STATE_FIELDS = ["speed", "body_slip_deg", "yaw_rate", "steering_wheel_deg", "road_wheel_deg", "ax_g", "ay_g",
                    "pedal", "brake", "motor_torque", "motor_rpm", "soc", "pack_temp", "lap_time", "lap",
                    "last_lap_time"]
_MESSAGES = {}


def numeric_message(name: str, fields) -> type:
    """A protobuf message of doubles (kmslap.<name>), built here (no .proto to ship)."""
    if name not in _MESSAGES:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
        fd = descriptor_pb2.FileDescriptorProto(name=f"kmslap_{name}.proto", package="kmslap", syntax="proto3")
        m = fd.message_type.add(name=name)
        for i, f in enumerate(fields, 1):
            m.field.add(name=f, number=i, type=descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
                        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL)
        pool = descriptor_pool.DescriptorPool()
        pool.Add(fd)
        _MESSAGES[name] = message_factory.GetMessageClass(pool.FindMessageTypeByName(f"kmslap.{name}"))
    return _MESSAGES[name]


def _sim_state_class():
    """The sim's ground truth (/sim/state)."""
    return numeric_message("SimState", SIM_STATE_FIELDS)


def sim_state(row, car, lap: float, lap_time: float, last_lap_time: float = 0.0):
    """/sim/state from a drive log row (a Series or a dict of its columns)."""
    g = row.get
    return _sim_state_class()(
        speed=float(g("speed", math.hypot(g("vx"), g("vy")))), body_slip_deg=math.degrees(float(g("beta", 0.0))),
        yaw_rate=float(g("r")), steering_wheel_deg=math.degrees(float(g("steering_wheel"))),
        road_wheel_deg=math.degrees(float(g("steering_wheel")) / car.steer_ratio), ax_g=float(g("ax")) / G,
        ay_g=float(g("ay")) / G, pedal=float(g("pedal")), brake=float(g("brake")),
        motor_torque=float(g("motor_torque")), motor_rpm=float(g("motor_rpm")), soc=float(g("soc")),
        pack_temp=float(g("pack_temp")), lap_time=float(lap_time), lap=float(lap), last_lap_time=float(last_lap_time))


def car_pose(x: float, y: float, psi: float, ns: int):
    """/sim/tf: map -> car."""
    from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
    tf = FrameTransform(parent_frame_id="map", child_frame_id="car")
    tf.timestamp.FromNanoseconds(ns)
    tf.translation.x, tf.translation.y = float(x), float(y)
    q = _quat("z", float(psi))
    tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w = (float(v) for v in q)
    return tf


def layout() -> dict:
    """A Foxglove layout for sim logs: 3D (following the car), map, the usual plots."""
    def plot(title, *paths):
        return {"paths": [{"value": p, "enabled": True, "timestampMethod": "receiveTime"} for p in paths],
                "showLegend": True, "legendDisplay": "floating", "foxglovePanelTitle": title}
    cfg = {
        "3D!sim": {"followTf": "car", "followMode": "follow-position",
                   # no frame axes / the map -> car line across the track
                   "scene": {"transforms": {"showLabel": False, "axisSize": 0, "lineWidth": 0}},
                   "transforms": {"frame:car": {"visible": False}, "frame:map": {"visible": False}},
                   "topics": {"/sim/track": {"visible": True}, "/sim/car": {"visible": True}},
                   "cameraState": {"distance": 25, "perspective": True, "phi": 55, "thetaOffset": 90, "targetOffset": [0, 0, 0]},
                   "foxglovePanelTitle": "car (sim)"},
        "map!sim": {"topicColors": {}, "disabledTopics": [], "foxglovePanelTitle": "where"},
        "Plot!speed": plot("speed (m/s) and body slip (deg)", "/sim/state.speed", "/sim/state.body_slip_deg"),
        "Plot!pedals": plot("Pedal Readings", "vcu_pedals_travel_data.vcu_apps1_travel", "vcu_pedals_travel_data.vcu_bse_travel"),
        "Plot!wheels": plot("wheel speed (rpm)", *[f"an1_{w}_wheelspeed_data.wheel_rpm" for w in ("fl", "fr", "rl", "rr")]),
        "Plot!loads": plot("tire load (N)", *[f"an1_{w}_loadcell_data.force_N" for w in ("fl", "fr", "rl", "rr")]),
        "Plot!brakes": plot("brake pressure (kPa), steering (deg)", "an1_front_brakepressure_data.pressure_kPa",
                            "an1_rear_brakepressure_data.pressure_kPa", "vcu_pedals_travel_data.steering_angle"),
        "Plot!torque": plot("torque (N m)", "m172_torque_and_timer_info_data.INV_Commanded_Torque",
                            "m172_torque_and_timer_info_data.INV_Torque_Feedback"),
        "Plot!power": plot("pack", "m167_voltage_info_data.INV_DC_Bus_Voltage", "msgid_0x6b1_data.Pack_Current",
                           "msgid_0x6b3_data.Pack_SOC"),
        "Plot!g": plot("g", "/sim/state.ax_g", "/sim/state.ay_g"),
        "Plot!vn": plot("VectorNav accel (m/s^2)", "vectornav_acceleration_data.vn_accel_x", "vectornav_acceleration_data.vn_accel_y"),
        "StateTransitions!vcu": {"paths": [{"value": "vcu_status_data.vcu_statemachine_state", "timestampMethod": "receiveTime"},
                                           {"value": "m170_internal_states_data.INV_VSM_State", "timestampMethod": "receiveTime"}]},
    }
    tree = {"direction": "row", "splitPercentage": 55,
            "first": {"direction": "column", "splitPercentage": 65, "first": "3D!sim",
                      "second": {"direction": "row", "first": "map!sim", "second": "StateTransitions!vcu"}},
            "second": {"direction": "row",
                       "first": {"direction": "column", "first": "Plot!speed",
                                 "second": {"direction": "column", "first": "Plot!pedals",
                                            "second": {"direction": "column", "first": "Plot!torque", "second": "Plot!power"}}},
                       "second": {"direction": "column", "first": "Plot!wheels",
                                  "second": {"direction": "column", "first": "Plot!loads",
                                             "second": {"direction": "column", "first": "Plot!brakes",
                                                        "second": {"direction": "column", "first": "Plot!g", "second": "Plot!vn"}}}}}}
    return {"configById": cfg, "globalVariables": {}, "userNodes": {}, "playbackConfig": {"speed": 1}, "layout": tree}


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def write(d: pd.DataFrame, track, out, car: Car = None, plan_line=None,
          start_ns: int | None = None, preroll_s: float = 5.0, seed: int = 0, lap_times=None,
          extra_events=()) -> dict:
    """Write a drive log (driving.drive's DataFrame) as the logger's MCAP plus /sim truth.
    start_ns: the log's start (default now); preroll_s: seconds parked with the inverter
    off before the run, as real logs begin; lap_times: the driver's start/finish
    crossings (s). extra_events: more (sim time s, topic, protobuf) to write (e.g. a
    HIL run's /hil topics). Returns {topic: message count}."""
    import cantools
    from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
    from mcap_protobuf.writer import Writer

    car = car or Car()
    folder = vehicle_dir(car)
    db = cantools.database.load_file(str(folder / "car.dbc"), strict=False)
    classes = _protos(folder)
    rates = sensors(car)
    start_ns = start_ns or time.time_ns()
    # the pre-roll: the first sample, parked, inverter off
    pre = d.iloc[[0] * max(int(preroll_s * 50), 1)].copy()
    pre["t"] = np.linspace(-preroll_s, -1e-3, len(pre))
    for k in ("pedal", "brake", "motor_torque", "motor_rpm", "bus_i", "vx", "vy", "r", "ax", "ay", "steering_wheel",
              *[f"hub_{x}_{w}" for x in ("torque", "cmd") for w in ("fl", "fr", "rl", "rr")]):
        if k in pre:
            pre[k] = 0.0
    if "torque_cmd" in pre:
        pre["torque_cmd"] = 0.0
    d = pd.concat([pre, d], ignore_index=True)
    sigs = sensor_signals(d, track, car, start_ns, seed)
    sigs.update(_states(db, d, car))
    t = d.t.to_numpy()
    rng = np.random.default_rng(seed + 1)
    events = []        # (time ns, topic, protobuf)
    dropped = {}
    for name, values in sigs.items():
        values = {k: (v.to_numpy() if isinstance(v, pd.Series) else v) for k, v in values.items()}
        msg = db.get_message_by_name(name)
        topic = msg.name.lower() + "_data"
        cls = classes[msg.name.lower()]
        rate = rates[name][0] if name in rates else 4.1        # state messages: the inverter's rate
        times = np.arange(t[0], t[-1], 1 / rate) + rng.uniform(0, 1 / rate)
        idx = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
        for k, ts in zip(idx, times):
            vals = {s: (v[k] if isinstance(v, np.ndarray) else v) for s, v in values.items()}
            vals = {s: (float(v) if isinstance(v, (np.floating, np.integer, np.bool_)) else v) for s, v in vals.items()}
            try:
                pb = _pack(cls, _wire(msg, vals))
            except (KeyError, ValueError, TypeError) as e:
                dropped.setdefault(topic, repr(e))   # the logger drops what it can't decode or pack
                continue
            events.append((start_ns + int((ts - t[0]) * 1e9), topic, pb))
    if dropped:
        # the real logger drops these silently; here it's a bug in a stand-in
        raise ValueError(f"messages that didn't encode: {dropped}")
    # ground truth for Foxglove: car pose, the car (steered wheels), track, location
    tf_t = np.arange(t[0], t[-1], 1 / 50)
    j = np.clip(np.searchsorted(t, tf_t), 0, len(t) - 1)
    for k in j:
        ns = start_ns + int((t[k] - t[0]) * 1e9)
        events.append((ns, "/sim/tf", car_pose(d.X.iloc[k], d.Y.iloc[k], d.psi.iloc[k], ns)))
        events.append((ns, "/sim/car", car_scene(car, float(d.steering_wheel.iloc[k]) / car.steer_ratio, ns)))
        if track.origin is not None and k % 2 == 0:
            lon, lat = track.lonlat(float(d.X.iloc[k]), float(d.Y.iloc[k]))
            fix = LocationFix(latitude=float(lat), longitude=float(lon), frame_id="car")
            fix.timestamp.FromNanoseconds(ns)
            events.append((ns, "/sim/location", fix))
    starts = np.r_[0.0, np.asarray(lap_times or [], float)]
    lap_no = np.searchsorted(starts, t, side="right") - 1
    lap_t = t - starts[np.maximum(lap_no, 0)]
    laps_done = np.diff(starts)
    for k in np.clip(np.searchsorted(t, np.arange(t[0], t[-1], 1 / 50)), 0, len(t) - 1):
        ns = start_ns + int((t[k] - t[0]) * 1e9)
        last = laps_done[lap_no[k] - 1] if lap_no[k] >= 1 else 0.0
        events.append((ns, "/sim/state", sim_state(d.iloc[k], car, lap_no[k], lap_t[k], last)))
    for ts, topic, pb in extra_events:
        events.append((start_ns + int((ts - t[0]) * 1e9), topic, pb))
    events.append((start_ns, "/sim/track", track_scene(track, car, plan_line)))
    events.sort(key=lambda e: e[0])
    counts = {}
    with open(out, "wb") as f:
        w = Writer(f)
        for ns, topic, pb in events:
            w.write_message(topic=topic, message=pb, log_time=ns, publish_time=ns)
            counts[topic] = counts.get(topic, 0) + 1
        source = ("KSU-MS/ksu-ms-dbc main @ ffa1f7d" if folder == DIR
                  else (DIR / "vehicles" / "SOURCE").read_text().splitlines()[0] + f", vehicle {folder.name}")
        w.add_metadata("kmslap_sim", {"track": track.name, "dbc": source,
                                      "sensors": json.dumps({k: v[2] for k, v in rates.items()}),
                                      "not_simulated": json.dumps(NOT_SIMULATED)})
        w.finish()
    return counts


def run(track_name: str = "ac18", laps: int = 2, grip: float = 0.85, margin: float = 0.35,
        step: float = 2.0, width: float | None = None, out: str = "sim.mcap", start: str | None = None,
        drivetrain: str = "rwd_emrax", soc: float = 1.0) -> dict:
    """Drive `laps` laps of the track's minimum-time plan (events/laptime, cached) and
    write the log, plus a Foxglove layout beside it (<out>.layout.json).
    start: the log's start time (ISO, e.g. 2025-06-13T14:00); default now.
    drivetrain: a car/drivetrain variant: hub motor cars plan and drive as kmslap laptime
    does (QP torque control, the brake pedal's regen travel, the inverters' command
    models), in their vehicle's DBC."""
    from kmslap.driving.drive import drive
    from kmslap.driving.follower import PathFollower
    from kmslap.events import track as T
    from kmslap.events.laptime import plan

    from kmslap.car.drivetrain import variant
    from kmslap.car.powertrain import PACKS, Pack
    from kmslap.driving.policies import RegenPolicy
    from kmslap.events.laptime import regen_power, regen_wheel_torque

    car = variant(drivetrain)
    hubs = car.hub_front or car.hub_rear
    track = T.TRACK_NAMES[track_name](width) if width is not None else T.TRACK_NAMES[track_name]()
    p = plan(track, grip, margin, step, car, regen=bool(hubs), soc=soc, label=f"{track.name} {drivetrain}")
    regen, kw = None, {}
    if hubs:
        regen = RegenPolicy(regen_travel=0.9)
        kw = dict(regen_travel=0.9, regen_wheel_torque=regen_wheel_torque(car, regen),
                  regen_power=regen_power(car, Pack(PACKS["cosmx600v"]), soc))
    f = PathFollower.for_plan(p, car=car, laps=laps, **kw)
    line = p.line
    res = drive(f, car, psi0=float(line.heading[0] - p.beta_ff[0]), X0=float(line.x[0]), Y0=float(line.y[0]),
                t_max=(laps + 1) * 4 * p.lap_time, log_period=0.005, regen=regen, soc0=soc,
                inverters="native" if hubs else None, torque_mode="qp",
                progress=f"{track.name} {drivetrain}: {laps} laps for the log, double-track model")
    start_ns = int(pd.Timestamp(start).value) if start else None
    counts = write(res.df, track, out, car, plan_line=line,
                   start_ns=start_ns, lap_times=f.lap_times)
    layout_path = str(out).rsplit(".", 1)[0] + ".layout.json"
    with open(layout_path, "w") as fh:
        json.dump(layout(), fh, indent=1)
    laps_s = np.diff([0.0] + f.lap_times)
    print(f"{track.name}: {len(laps_s)} laps {np.round(laps_s, 2).tolist()} -> {out} ({sum(counts.values())} messages, "
          f"{len(counts)} topics); Foxglove layout {layout_path}")
    return counts
