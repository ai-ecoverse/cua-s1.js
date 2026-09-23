#!/usr/bin/env bash
# Build, verify and package cua-s1-4b for the browser at an exact Hub commit:
#
#   ./build_4b_model.sh [--multimodal] [cua-ai/cua-s1-4b-0.1[@rev]]
#
# pin -> merge the LoRA in fp32 -> fp32 + int8 WebGPU decoder graphs -> 32 MB shards -> FourBModel fixtures -> parity
# of the fp32 graphs -> package (with the int8 variant's parity) into ../public/models/<name>[-multimodal].
# --multimodal takes the screenshot adapter: it also exports the vision tower (fp32 and int8), splices image_embeds
# into the decoders, and renders the fixture screenshots with DejaVu Sans, as the generator does on Linux.
# Peaks near 45 GB of disk (60 GB multimodal).
set -euo pipefail
cd "$(dirname "$0")"
mm=0; [ "${1:-}" = --multimodal ] && { mm=1; shift; }
repo=${1:-cua-ai/cua-s1-4b-0.1}
run=$(uv run --group four-b python -c "from four_b import pin; print(pin('$repo'))" 2>/dev/null | tail -1)
name=$(basename "${run%@*}"); [ $mm = 1 ] && name=$name-multimodal
modality=text; image=(); [ $mm = 1 ] && { modality=multimodal; image=(--image-token-id 248056); }
out=build/$name
fx=../fixtures/$name.json
log() { echo "[$name] $*"; }
log "pinned $run"
log "merge"
rm -rf "$out"
uv run --group four-b python -m four_b.merge --adapter "$run" --modality $modality --out "$out"
log "build"
./build_4b.sh "$out" fp32-cpu q8f32-webgpu
log "shard"
uv run --group four-b python -m four_b.postprocess --src "$out/onnx-q8f32-webgpu" --out "$out/web-q8f32" ${image[@]+"${image[@]}"} | tail -1 | cut -c1-120
rm -rf "$out/onnx-q8f32-webgpu"
ref=$out/onnx-fp32-cpu; vision=()
if [ $mm = 1 ]; then
  uv run --group four-b python -m four_b.postprocess --src "$out/onnx-fp32-cpu" --out "$out/fp32" --embed keep --shard-mb 2000 "${image[@]}" | tail -1 | cut -c1-120
  rm -rf "$out/onnx-fp32-cpu"; ref=$out/fp32
  log "vision tower"
  uv run --group four-b python -m four_b.vision --merged "$out/merged" --out "$out/vision" | grep -v "INFO\|Warning" | tail -6
  if [ ! -f build/fonts/DejaVuSans.ttf ]; then
    mkdir -p build/fonts
    curl -sSL -o build/fonts/dejavu.zip https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip
    unzip -qo -j build/fonts/dejavu.zip "*/ttf/DejaVuSans.ttf" "*/ttf/DejaVuSans-Bold.ttf" -d build/fonts && rm build/fonts/dejavu.zip
  fi
  vision=(--vision "$out/vision/fp32.onnx" --merged "$out/merged")
fi
log "fixtures (FourBModel, fp32)"
per_app=3; [ $mm = 1 ] && per_app=2
uv run --group four-b python -m four_b.fixtures --adapter "$run" --modality $modality --per-app $per_app --out "$fx" | tail -1
log "parity (fp32 graphs)"
uv run --group four-b python -m four_b.parity --model "$ref/model.onnx" --head "$out/head.safetensors" --fixtures "$fx" ${vision[@]+"${vision[@]}"}
rm -rf "$ref"
log "package"
pkg=(); [ $mm = 1 ] && pkg=(--vision "$out/vision/web" --merged "$out/merged")
uv run --group four-b python -m four_b.package --build "$out" --out "../public/models/$name" --variant q8f32=web-q8f32 --fixtures "$fx" ${pkg[@]+"${pkg[@]}"} | tail -2
rm -rf "$out/merged"
log "done"
