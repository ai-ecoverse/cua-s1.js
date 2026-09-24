"""Shrink a builder export for the browser without touching the transformer body (from kev.js,
kev_web_export/postprocess.py: the same Qwen3.5 builder output).

- The embedding table is the single largest tensor (vocab 248k x hidden) and the builder only quantizes it to int4
  (GatherBlockQuantized). --embed int8 stores it as int8 with one scale per row, dequantized after the lookup with
  standard ops (Gather, Cast, Mul), so every execution provider can run it.
- The rotary cos/sin caches are sized for the base model's 262k context. A cua-s1-4b prompt is an accessibility
  tree and at most 26 options, a few thousand tokens, so --rope-positions trims them.
- --image-token-id adds an `image_embeds` input [images' merged patches, hidden] for the multimodal adapter: each
  <|image_pad|> position of input_ids takes the next row, in order, instead of the token's embedding. That is
  transformers' masked_scatter of the vision features into inputs_embeds. Without an image, pass one row of zeros."""
import argparse, os, shutil
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="builder output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed", choices=["keep", "int8"], default="int8")
    ap.add_argument("--rope-positions", type=int, default=8192)
    ap.add_argument("--image-token-id", type=int, help="splice image_embeds in at this token id (multimodal)")
    ap.add_argument("--shard-mb", type=int, default=32,
                    help="max size of one external-data file. Small shards keep every request well inside proxy and "
                         "CDN response caps (bb connect cuts a response at 34.5 MiB) and make a failed download cheap "
                         "to retry; browsers also cap a single buffer near 2 GB")
    a = ap.parse_args()
    m = onnx.load(f"{a.src}/model.onnx", load_external_data=True)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    (embed,) = [n for n in g.node if n.op_type == "Gather" and n.input[0] == "model.embed_tokens.weight"]
    embedded = embed.output[0]   # the embedding lookup's output, whatever computes it below
    hidden = inits["model.embed_tokens.weight"].dims[1]

    for name in ("cos_cache", "sin_cache"):
        t = inits[name]; arr = numpy_helper.to_array(t)
        if arr.shape[0] > a.rope_positions:
            t.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(arr[: a.rope_positions]), name))

    if a.embed == "int8":
        (node,) = [n for n in g.node if n.op_type == "Gather" and n.input[0] == "model.embed_tokens.weight"]
        w = numpy_helper.to_array(inits["model.embed_tokens.weight"])
        wf = w.astype(np.float32)
        scale = np.maximum(np.abs(wf).max(axis=1), 1e-12) / 127.0
        q = np.clip(np.rint(wf / scale[:, None]), -127, 127).astype(np.int8)
        err = float(np.abs(q.astype(np.float32) * scale[:, None] - wf).max())
        g.initializer.remove(inits["model.embed_tokens.weight"])

        # One initializer per hidden-dimension slice: an embedding table is a single tensor of hundreds of MB, and an
        # ONNX initializer cannot be split across external-data files. Slicing the columns keeps every file small while
        # the result is bit-identical: each slice is gathered with the same ids and the outputs are concatenated.
        vocab, hidden = q.shape
        slices = max(1, -(-q.nbytes // (a.shard_mb * 1_000_000)))
        width = -(-hidden // slices)
        out, ids, p = node.output[0], node.input[1], "/model/embed_tokens"
        io = helper.np_dtype_to_tensor_dtype(w.dtype)
        new_nodes, cast_outputs = [], []
        for k, start in enumerate(range(0, hidden, width)):
            part = np.ascontiguousarray(q[:, start : start + width])
            g.initializer.append(numpy_helper.from_array(part, f"model.embed_tokens.weight_Q8_{k}"))
            new_nodes += [helper.make_node("Gather", [f"model.embed_tokens.weight_Q8_{k}", ids], [f"{p}/GatherQ8_{k}/output_0"], name=f"{p}/GatherQ8_{k}"),
                          helper.make_node("Cast", [f"{p}/GatherQ8_{k}/output_0"], [f"{p}/CastQ8_{k}/output_0"], name=f"{p}/CastQ8_{k}", to=io)]
            cast_outputs.append(f"{p}/CastQ8_{k}/output_0")
        g.initializer.append(numpy_helper.from_array(scale.astype(w.dtype), "model.embed_tokens.weight_scales"))
        joined = cast_outputs[0] if len(cast_outputs) == 1 else f"{p}/Concat/output_0"
        if len(cast_outputs) > 1:
            new_nodes.append(helper.make_node("Concat", cast_outputs, [joined], name=f"{p}/Concat", axis=-1))
        new_nodes += [helper.make_node("Gather", ["model.embed_tokens.weight_scales", ids], [f"{p}/GatherScale/output_0"], name=f"{p}/GatherScale"),
                      helper.make_node("Unsqueeze", [f"{p}/GatherScale/output_0", f"{p}/axes_last"], [f"{p}/Unsqueeze/output_0"], name=f"{p}/Unsqueeze"),
                      helper.make_node("Mul", [joined, f"{p}/Unsqueeze/output_0"], [out], name=f"{p}/MulScale")]
        g.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), f"{p}/axes_last"))
        i = list(g.node).index(node); g.node.remove(node)
        for k, n in enumerate(new_nodes): g.node.insert(i + k, n)
        print(f"embedding -> int8 per-row in {len(cast_outputs)} column slices (max abs error {err:.2e})")

    if a.image_token_id is not None:
        splice_image_embeds(g, embedded, a.image_token_id, hidden)
        print(f"image_embeds spliced in at token {a.image_token_id}")

    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.src):
        if f.endswith((".json", ".jinja")): shutil.copy(f"{a.src}/{f}", a.out)
    for f in os.listdir(a.out):
        if f.startswith("model.onnx.data"): os.remove(f"{a.out}/{f}")
    files = save_sharded(m, a.out, a.shard_mb * 1_000_000)
    print(f"{a.out}: " + ", ".join(f"{f} {os.path.getsize(f'{a.out}/{f}') / 1e6:.0f} MB" for f in files))


def splice_image_embeds(g, embedded, token_id, hidden):
    (producer,) = [n for n in g.node if embedded in n.output]
    text = f"{embedded}_text"
    producer.output[list(producer.output).index(embedded)] = text
    ids = next(i for i in g.input if i.name == "input_ids")
    io = next(o for o in g.output if o.name == "hidden_states").type.tensor_type.elem_type
    g.input.append(helper.make_tensor_value_info("image_embeds", io, ["image_tokens", hidden]))
    p = "/model/image_embeds"
    g.initializer.extend([numpy_helper.from_array(np.array(token_id, np.int64), f"{p}/token_id"),
                          numpy_helper.from_array(np.array(1, np.int64), f"{p}/one"),
                          numpy_helper.from_array(np.array(0, np.int64), f"{p}/zero"),
                          numpy_helper.from_array(np.array(1, np.int64), f"{p}/axis"),
                          numpy_helper.from_array(np.array([-1], np.int64), f"{p}/last")])
    nodes = [
        helper.make_node("Equal", [ids.name, f"{p}/token_id"], [f"{p}/mask"], name=f"{p}/Equal"),                  # [B, S]
        helper.make_node("Cast", [f"{p}/mask"], [f"{p}/mask_i"], name=f"{p}/Cast", to=TensorProto.INT64),
        helper.make_node("CumSum", [f"{p}/mask_i", f"{p}/axis"], [f"{p}/count"], name=f"{p}/CumSum"),               # 1, 2, ... at image tokens
        helper.make_node("Sub", [f"{p}/count", f"{p}/one"], [f"{p}/row"], name=f"{p}/Sub"),
        helper.make_node("Max", [f"{p}/row", f"{p}/zero"], [f"{p}/row0"], name=f"{p}/Max"),                         # text tokens before the image
        helper.make_node("Gather", ["image_embeds", f"{p}/row0"], [f"{p}/rows"], name=f"{p}/Gather", axis=0),     # [B, S, hidden]
        helper.make_node("Unsqueeze", [f"{p}/mask", f"{p}/last"], [f"{p}/mask3"], name=f"{p}/Unsqueeze"),
        helper.make_node("Where", [f"{p}/mask3", f"{p}/rows", text], [embedded], name=f"{p}/Where"),
    ]
    i = list(g.node).index(producer) + 1
    for k, n in enumerate(nodes): g.node.insert(i + k, n)


def save_sharded(m, out, max_bytes, threshold=1024):
    """Write initializers over `threshold` bytes to model.onnx.data, model.onnx.data_1, ... (each <= max_bytes unless a
    single tensor is larger), then the graph itself to model.onnx."""
    from onnx.external_data_helper import set_external_data
    files, fh, off = [], None, 0
    for t in m.graph.initializer:
        raw = t.raw_data if t.HasField("raw_data") else numpy_helper.from_array(numpy_helper.to_array(t), t.name).raw_data
        if len(raw) <= threshold: continue
        if fh is None or (off and off + len(raw) > max_bytes):
            if fh: fh.close()
            files.append("model.onnx.data" if not files else f"model.onnx.data_{len(files)}")
            fh, off = open(f"{out}/{files[-1]}", "wb"), 0
        fh.write(raw)
        nt = TensorProto(name=t.name, data_type=t.data_type, dims=list(t.dims), raw_data=raw)
        set_external_data(nt, location=files[-1], offset=off, length=len(raw))
        nt.data_location = TensorProto.EXTERNAL; nt.ClearField("raw_data")
        t.CopyFrom(nt)
        off += len(raw)
    if fh: fh.close()
    with open(f"{out}/model.onnx", "wb") as f: f.write(m.SerializeToString())
    return files


if __name__ == "__main__":
    main()
