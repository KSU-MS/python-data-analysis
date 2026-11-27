# cardata/mcap: the car logger's format

What `kmslap mcap` (`kmslap/reporting/mcap_log.py`) needs to write simulated logs the
team's tools read like real ones, for the current car:

| File | What | From |
|---|---|---|
| `car.dbc` | the car's CAN database | `KSU-MS/ksu-ms-dbc` `main` @ `ffa1f7d` (1 Nov 2026), built by its flake |
| `hytech.proto`, `hytech.bin` | the logger's protobuf schema for that DBC (and its descriptor set) | `KSU-MS/ksu_daq`'s `py_dbc_proto_gen/dbc_to_proto.py` on `car.dbc`, then `protoc --include_imports --descriptor_set_out=hytech.bin hytech.proto`, as its Nix build does |
| `log_profile.json` | message rates, VectorNav noise, pedal raw ranges | measured from twelve real logs from 2025-06-13 (the car stationary, wheels off the ground), under that day's message names |

The logger (`KSU-MS/ksu_daq`, a fork of HyTech's `data_acq`): CAN frame → cantools
decode with the DBC (value tables as names) → `pack_protobuf_msg` into the generated
message (lowercased message name; field names are signal names without spaces or
parentheses; value tables become strings, scaled signals floats, 1-bit bools) →
`mcap_protobuf` `Writer` to topic `<message>_data`, stamped with the Pi's clock.

The June 2025 logs were made with an older DBC (`ancient-pkgs` @ `64a9257`: the proto
generated from it matches those logs field for field). The current DBC renamed most of
the VCU and VectorNav messages, added the AN1 boards (wheel speeds, load cells, shock
pots, brake pressures, steering pot, tire / motor / coolant temperatures) and fixed the
old field definitions (signed pack current, 16-bit cell voltages). Those logs' temporary
quirks (inverter torque reported negative, the VectorNav mounted on its side) are fixed
on the car and not reproduced.

To move to a newer DBC:

```sh
nix build github:KSU-MS/ksu-ms-dbc/<rev>                 # result/car.dbc
python dbc_to_proto.py result                           # ksu_daq/py_dbc_proto_gen
protoc --include_imports --descriptor_set_out=hytech.bin hytech.proto
```

then update `SENSORS` in `mcap_log.py` for renamed messages (its test checks every
mapped message is written in this schema).
