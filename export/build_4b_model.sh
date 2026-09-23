#!/usr/bin/env bash
# Build, verify and package cua-s1-4b for the browser at an exact Hub commit:
#
#   ./build_4b_model.sh [cua-ai/cua-s1-4b-0.1[@rev]]
#
# pin -> merge the text LoRA in fp32 -> fp32 + int8 WebGPU graphs -> 32 MB shards -> FourBModel fixtures -> parity of
# the fp32 graph -> package (with the int8 variant's parity) into ../public/models/<name>. Peaks near 45 GB of disk.
set -euo pipefail
cd "$(dirname "$0")"
repo=${1:-cua-ai/cua-s1-4b-0.1}
run=$(uv run --group four-b python -c "from four_b import pin; print(pin('$repo'))" 2>/dev/null | tail -1)
name=$(basename "${run%@*}")
out=build/$name
log() { echo "[$name] $*"; }
log "pinned $run"
log "merge"
rm -rf "$out"
uv run --group four-b python -m four_b.merge --adapter "$run" --out "$out"
log "build"
./build_4b.sh "$out" fp32-cpu q8f32-webgpu
rm -rf "$out/merged"
log "shard"
uv run --group four-b python -m four_b.postprocess --src "$out/onnx-q8f32-webgpu" --out "$out/web-q8f32" | tail -1 | cut -c1-120
rm -rf "$out/onnx-q8f32-webgpu"
log "fixtures (FourBModel, fp32)"
uv run --group four-b python -m four_b.fixtures --adapter "$run" --out "../fixtures/$name.json" | tail -1
log "parity (fp32 graph)"
uv run --group four-b python -m four_b.parity --model "$out/onnx-fp32-cpu/model.onnx" --head "$out/head.safetensors" --fixtures "../fixtures/$name.json"
rm -rf "$out/onnx-fp32-cpu"
log "package"
uv run --group four-b python -m four_b.package --build "$out" --out "../public/models/$name" --variant q8f32=web-q8f32 --fixtures "../fixtures/$name.json" | tail -2
log "done"
