#!/usr/bin/env bash
# Build, verify and package cua-s1-4b for the browser at an exact Hub commit:
#
#   ./build_4b_model.sh [--multimodal] [cua-ai/cua-s1-4b-0.2[@rev]]
#
# pin -> merge the LoRA in fp32 -> fp32 + int8 WebGPU decoder graphs -> 32 MB shards -> FourBModel fixtures on GUI-360's
# test split -> parity of the fp32 graphs -> package (with the int8 variant's parity) into ../public/models/<name>[-multimodal].
# --multimodal takes the screenshot adapter: it also exports the vision tower (fp32 and fp16), splices image_embeds
# into the decoders, and extracts GUI-360's test screenshots (a 6.3 GB download, once).
# Peaks near 45 GB of disk (60 GB multimodal).
set -euo pipefail
cd "$(dirname "$0")"
mm=0; [ "${1:-}" = --multimodal ] && { mm=1; shift; }
repo=${1:-cua-ai/cua-s1-4b-0.2}
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
  if [ ! -d build/gui360/image ]; then
    log "GUI-360 test screenshots"
    tgz=$(uv run --group four-b python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('vyokky/GUI-360', 'test/image.tar.gz', repo_type='dataset'))" | tail -1)
    mkdir -p build/gui360 && tar xzf "$tgz" -C build/gui360 --include 'image/*/in_app/success/*/action_step*.png' --exclude '*_annotated.png'
  fi
  vision=(--vision "$out/vision/fp32.onnx" --merged "$out/merged")
fi
log "fixtures (FourBModel, fp32)"
uv run --group four-b python -m four_b.fixtures --adapter "$run" --modality $modality --out "$fx" | tail -1
log "parity (fp32 graphs)"
uv run --group four-b python -m four_b.parity --model "$ref/model.onnx" --head "$out/head.safetensors" --fixtures "$fx" ${vision[@]+"${vision[@]}"}
rm -rf "$ref"
log "package"
pkg=(); [ $mm = 1 ] && pkg=(--vision "$out/vision/web" --merged "$out/merged")
uv run --group four-b python -m four_b.package --build "$out" --out "../public/models/$name" --variant q8f32=web-q8f32 --fixtures "$fx" ${pkg[@]+"${pkg[@]}"} | tail -2
rm -rf "$out/merged"
log "done"
