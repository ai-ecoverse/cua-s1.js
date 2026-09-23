#!/usr/bin/env bash
# Build ONNX graphs of a merged cua-s1-4b checkpoint (four_b.merge output): build_4b.sh build/cua-s1-4b-0.2 [variant...]
# variants: fp32-cpu (the parity reference), q8f32-webgpu (int8 weights, fp32 activations: what the browser runs)
set -euo pipefail
cd "$(dirname "$0")"
dir=$1; shift
common=(exclude_lm_head=true exclude_mtp=true)   # hidden states out: the head is the letters' embedding rows
for v in ${*:-fp32-cpu q8f32-webgpu}; do
  case $v in
    fp32-cpu)     args=(-p fp32 -e cpu --extra_options "${common[@]}") ;;
    q8f32-webgpu) args=(-p int8 -e webgpu --extra_options "${common[@]}" use_webgpu_fp32=true) ;;
    *) echo "unknown variant $v" >&2; exit 2 ;;
  esac
  echo "== $v"
  # the builder can abort in process teardown on macOS (libc++ "recursive_mutex lock failed") after everything is
  # written, so success is judged by genai_config.json, which it writes last
  rm -rf "$dir/onnx-$v"
  uv run --group four-b python -m onnxruntime_genai.models.builder -i "$dir/merged" -o "$dir/onnx-$v" -c build/cache "${args[@]}" > "$dir/onnx-$v.log" 2>&1 || true
  [ -f "$dir/onnx-$v/genai_config.json" ] || { tr '\r' '\n' < "$dir/onnx-$v.log" | tail -20; exit 1; }
  du -sh "$dir/onnx-$v/model.onnx.data"
done
