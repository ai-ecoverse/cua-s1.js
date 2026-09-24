"""Export cua-s1-4b's vision tower (Qwen3.5's ViT and patch merger, with the multimodal adapter merged) to ONNX.

    uv run python -m four_b.vision --merged build/cua-s1-4b-0.2-mm/merged --out build/cua-s1-4b-0.2-mm/vision

Qwen3_5VisionModel.forward derives everything that depends on the image's size from grid_thw: the bilinear
resampling of the learned 48x48 position table, the 2D rotary angles, and the packed-sequence boundaries. Those are
data-dependent loops, so here they are inputs instead, computed by the caller (vision_inputs below, and its port in
src/four-b-vision.ts). What remains is plain tensor math over one image's patches in spatial-merge-block order:

    patches [P, 1536] -> patch projection (the Conv3d, whose kernel equals its stride, as a MatMul)
                      + sum_k pos_embed[pos_idx[:, k]] * pos_w[:, k]
                      -> 24 blocks (full attention within the image, 2D rotary from cos/sin)
                      -> merger -> image_embeds [P/4, 2560]

`Core` is checked against transformers' own `visual(pixel_values, grid_thw)` before export, and the ONNX graph
against `Core`."""
import argparse, json, os
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file

INPUTS = ["patches", "pos_idx", "pos_w", "cos", "sin"]


def load_visual(merged: str):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    cfg = AutoConfig.from_pretrained(merged).vision_config
    cfg._attn_implementation = "eager"
    v = Qwen3_5VisionModel(cfg).float().eval()
    sd = {k[len("model.visual."):]: t for k, t in load_file(f"{merged}/model.safetensors").items() if k.startswith("model.visual.")}
    missing, unexpected = v.load_state_dict(sd, strict=False)
    if [m for m in missing if not m.endswith("inv_freq")] or unexpected:
        raise SystemExit(f"vision weights: missing {missing}, unexpected {unexpected}")
    return v


def vision_inputs(v, grid_h: int, grid_w: int):
    """The size-dependent inputs for one image of grid_h x grid_w patches, from transformers' own helpers."""
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    thw = torch.tensor([[1, grid_h, grid_w]])
    idx, w = get_vision_interpolation_indices_and_weights(thw, num_grid_per_side=v.num_grid_per_side, mode=v.interpolation_mode,
                                                          align_corners=v.interpolation_align_corners, spatial_merge_size=v.spatial_merge_size)
    pos = get_vision_position_ids(thw, v.spatial_merge_size)
    cos, sin = v.rotary_pos_emb(torch.zeros(1), pos)
    return idx, w.float(), cos.float(), sin.float()


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Core(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v
        p = v.patch_embed.proj
        self.patch_w = nn.Parameter(p.weight.detach().reshape(p.weight.shape[0], -1).clone())
        self.patch_b = nn.Parameter(p.bias.detach().clone())

    def forward(self, patches, pos_idx, pos_w, cos, sin):
        v = self.v
        h = patches @ self.patch_w.T + self.patch_b
        h = h + (v.pos_embed(pos_idx) * pos_w[:, :, None]).sum(1)
        cos, sin = cos[:, None, :], sin[:, None, :]
        for blk in v.blocks:
            a = blk.attn
            P = h.shape[0]
            q, k, val = a.qkv(blk.norm1(h)).reshape(P, 3, a.num_heads, -1).permute(1, 2, 0, 3).unbind(0)   # [heads, P, d]
            q = q * cos.transpose(0, 1) + rotate_half(q) * sin.transpose(0, 1)
            k = k * cos.transpose(0, 1) + rotate_half(k) * sin.transpose(0, 1)
            att = torch.softmax((q @ k.transpose(-1, -2)) * a.scaling, dim=-1) @ val                       # full attention
            h = h + a.proj(att.transpose(0, 1).reshape(P, -1))
            h = h + blk.mlp(blk.norm2(h))
        return v.merger(h)


def strip_traces(path):
    import onnx
    m = onnx.load(path, load_external_data=True)
    for node in m.graph.node:
        keep = [p for p in node.metadata_props if p.key != "pkg.torch.onnx.stack_trace"]
        del node.metadata_props[:]; node.metadata_props.extend(keep)
    return m


def half_weights(src: str, out_dir: str, shard_mb: int = 32):
    """The browser graph: weights stored as fp16 and cast to fp32 when the session loads, compute in fp32. Measured
    on the fixture screenshots, int8 MatMulNBits weights (blocks of 32, as the builder quantizes the decoder) put
    the image embeddings 4.6% off in RMS, which no single group of layers accounts for, and doubled the decoder's
    own int8 error in the letters' probabilities; fp16 weights stay within 0.2% for 1.7x the bytes (670 MB)."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from .postprocess import save_sharded
    m = onnx.load(src, load_external_data=True)
    g = m.graph
    for t in list(g.initializer):
        if t.data_type != TensorProto.FLOAT or np.prod(t.dims) < 65536: continue
        name, a = t.name, numpy_helper.to_array(t)
        g.initializer.remove(t)
        g.initializer.append(numpy_helper.from_array(a.astype(np.float16), f"{name}_f16"))
        g.node.insert(0, helper.make_node("Cast", [f"{name}_f16"], [name], name=f"{name}/Cast", to=TensorProto.FLOAT))
    os.makedirs(out_dir, exist_ok=True)
    save_sharded(m, out_dir, shard_mb * 1_000_000)
    return f"{out_dir}/model.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--web-only", action="store_true", help="just (re)write the browser graph from fp32.onnx")
    a = ap.parse_args()
    if a.web_only:
        print(f"-> {half_weights(f'{a.out}/fp32.onnx', f'{a.out}/web')}"); return
    v = load_visual(a.merged)
    core = Core(v).eval()

    # check the rewrite against the unmodified module on processor output for real screen-sized images
    from transformers import AutoImageProcessor
    from PIL import Image
    proc = AutoImageProcessor.from_pretrained(a.merged)
    rng = np.random.default_rng(0)
    for w, h in [(760, 632), (768, 1024), (500, 300)]:
        img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        out = proc(images=[img], return_tensors="pt")
        (t, gh, gw), = out["image_grid_thw"].tolist()
        with torch.no_grad():
            ref = v(out["pixel_values"].float(), grid_thw=out["image_grid_thw"]).pooler_output
            got = core(out["pixel_values"].float(), *vision_inputs(v, gh, gw))
        print(f"{w}x{h} -> grid {gh}x{gw}: Core vs visual max |d| {float((ref - got).abs().max()):.2e} (|ref| max {float(ref.abs().max()):.1f})")

    os.makedirs(a.out, exist_ok=True)
    img = Image.fromarray(rng.integers(0, 256, (632, 760, 3), dtype=np.uint8))
    out = proc(images=[img], return_tensors="pt")
    (_, gh, gw), = out["image_grid_thw"].tolist()
    sample = (out["pixel_values"].float(), *vision_inputs(v, gh, gw))
    from torch.export import Dim
    P = 4 * Dim("merged", min=1, max=16384)   # the merger folds 2x2 blocks of consecutive patches
    path = f"{a.out}/fp32.onnx"
    with torch.enable_grad():
        program = torch.onnx.export(core, sample, dynamo=True, opset_version=a.opset, input_names=INPUTS, output_names=["image_embeds"],
                                    dynamic_shapes={"patches": {0: P}, "pos_idx": {0: P}, "pos_w": {0: P}, "cos": {0: P}, "sin": {0: P}})
    program.save(path, external_data=True)
    import onnx
    onnx.save(strip_traces(path), path, save_as_external_data=True, location="fp32.onnx.data")

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for w, h in [(760, 632), (1280, 800)]:
        img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        out = proc(images=[img], return_tensors="pt")
        (_, gh, gw), = out["image_grid_thw"].tolist()
        ins = (out["pixel_values"].float(), *vision_inputs(v, gh, gw))
        with torch.no_grad(): ref = core(*ins).numpy()
        got = sess.run(["image_embeds"], {k: t.numpy() for k, t in zip(INPUTS, ins)})[0]
        print(f"ONNX {w}x{h}: max |d| vs Core {float(np.abs(ref - got).max()):.2e}")
    json.dump({"inputs": INPUTS, "outputs": ["image_embeds"], "patch_size": v.patch_size, "merge_size": v.spatial_merge_size,
               "temporal_patch_size": v.patch_embed.temporal_patch_size, "hidden_size": v.config.hidden_size,
               "num_heads": v.config.num_heads, "num_grid_per_side": v.num_grid_per_side,
               "rope_theta": v.config.rope_parameters["rope_theta"], "image_mean": proc.image_mean, "image_std": proc.image_std,
               "min_pixels": proc.size["shortest_edge"], "max_pixels": proc.size["longest_edge"]},
              open(f"{a.out}/vision.json", "w"), indent=2)
    print(f"-> {path}; browser (fp16 weights) -> {half_weights(path, f'{a.out}/web')}")


if __name__ == "__main__":
    main()
