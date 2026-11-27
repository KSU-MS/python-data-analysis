#!/usr/bin/env bash
# Each vehicle's DBC and the logger's protobuf schema for it, into vehicles/<name>/
# (car.dbc, hytech.proto, hytech.bin): ksu-ms-dbc's vehicle DBCs (gen_tool/vehicles.json),
# through ksu_daq's dbc_to_proto.py and protoc, as ksu_daq's Nix build does.
#
#   cardata/mcap/build_vehicles.sh [ksu-ms-dbc checkout] [ksu_daq checkout]
#
# The checkouts default to ~/dev/ksu-ms-dbc and ~/dev/ksu_daq; vehicles/SOURCE records
# the revisions.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dbc=$(cd "${1:-$HOME/dev/ksu-ms-dbc}" && pwd)
daq=$(cd "${2:-$HOME/dev/ksu_daq}" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

nix build "$dbc#default" -o "$work/result"
for f in "$work"/result/vehicles/car_*.dbc; do
    v=$(basename "$f" .dbc)
    v=${v#car_}
    mkdir -p "$work/$v" "$here/vehicles/$v"
    cp "$f" "$work/$v/car.dbc"
    (cd "$work/$v" && nix-shell -p python3Packages.cantools protobuf --run \
        "python3 '$daq/py_dbc_proto_gen/dbc_to_proto.py' . && protoc --include_imports --descriptor_set_out=hytech.bin hytech.proto")
    cp "$work/$v/car.dbc" "$work/$v/hytech.proto" "$work/$v/hytech.bin" "$here/vehicles/$v/"
    echo "$v: $(grep -c '^BO_ ' "$work/$v/car.dbc") messages"
done
{
    echo "ksu-ms-dbc $(git -C "$dbc" rev-parse --abbrev-ref HEAD) @ $(git -C "$dbc" rev-parse --short HEAD)$(git -C "$dbc" diff --quiet || echo ' (dirty)')"
    echo "ksu_daq $(git -C "$daq" rev-parse --abbrev-ref HEAD) @ $(git -C "$daq" rev-parse --short HEAD)"
    echo "built $(date -u +%Y-%m-%d) by cardata/mcap/build_vehicles.sh"
} > "$here/vehicles/SOURCE"
cat "$here/vehicles/SOURCE"
