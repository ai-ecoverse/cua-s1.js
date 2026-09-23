"""Export a cua-s1 checkpoint to ONNX for the browser and verify it against PyTorch.

    uv run python export.py --out ../public/models/cua-s1-forms

1. Pin the Hugging Face repo to a commit (checkpoints can be republished under the same id) and download the
   safetensors + JSON sidecar at that commit.
2. Load it with cua_s1's own loader, which validates the format and the SHA-256 tensor signature.
3. Export the unmodified model to ONNX with dynamic batch, context, option and option-token axes. The graph takes
   the tensors ByteCollator produces and returns per-option logits and probabilities.
4. Export a second graph from the same modules for the case every element shares one option list (a form plan): the
   options are encoded once and broadcast to every row, instead of once per row.
5. Score synthetic episodes (cua_s1.synth, the training generator) with PyTorch and both graphs, and write the
   PyTorch outputs as parity fixtures for the JS runtime.
"""
import argparse, hashlib, json, os, shutil
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from cua_s1.model import load_checkpoint
from cua_s1.synth import episode_rows

INPUTS = ["context_ids", "context_mask", "option_ids", "option_token_mask", "option_mask"]
SHARED_INPUTS = INPUTS[:4]   # option_ids / option_token_mask without the batch axis; every option is live


class Exported(torch.nn.Module):
    """TinyTransformerScorer.forward takes a dict; ONNX wants positional tensors."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context_ids, context_mask, option_ids, option_token_mask, option_mask):
        logits = self.model({"context_ids": context_ids, "context_mask": context_mask, "option_ids": option_ids,
                             "option_token_mask": option_token_mask, "option_mask": option_mask})
        return logits, logits.softmax(-1)


class SharedOptions(torch.nn.Module):
    """TinyTransformerScorer.forward with one option list for the whole batch. Same modules and arithmetic; the
    option encoder runs over [options, tokens] once instead of [batch * options, tokens]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context_ids, context_mask, option_ids, option_token_mask):
        m = self.model
        safe_context_mask = context_mask.clone()
        safe_context_mask[:, 0] = True
        context = m.encoder(m._embed(context_ids), src_key_padding_mask=~safe_context_mask)
        safe_mask = option_token_mask.clone()
        safe_mask[:, 0] = True
        hidden = m.option_encoder(m._embed(option_ids), src_key_padding_mask=~safe_mask)
        weights = option_token_mask.unsqueeze(-1).float()
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        options = pooled.unsqueeze(0).expand(context_ids.shape[0], -1, -1)
        logits = m.head(context, context_mask, options, options.new_ones(options.shape[:2], dtype=torch.bool))
        return logits, logits.softmax(-1)


def collate(collator, rows):
    from cua_s1.model import ChoiceExample
    batch = collator([ChoiceExample(context=r["context"], options=r["options"], label=r.get("label", 0)) for r in rows])
    return [batch[k] for k in INPUTS]


def collate_shared(collator, rows):
    """Rows that share one option list: the options' tensors from a one-row batch, the contexts from all of them."""
    assert len({tuple(r["options"]) for r in rows}) == 1
    context_ids, context_mask = collate(collator, rows)[:2]
    option_ids, option_token_mask = collate(collator, rows[:1])[2:4]
    return [context_ids, context_mask, option_ids[0], option_token_mask[0]]


def export(module, sample, names, dynamic_shapes, path, opset):
    # torch.export-based exporter: the legacy tracer bakes the sample's sequence length into nn.MultiheadAttention's
    # reshapes. Grad enabled + parameters requiring grad keeps nn.TransformerEncoderLayer off its fused inference
    # fast path (torch._transformer_encoder_layer_fwd); eval() still disables dropout.
    with torch.enable_grad():
        program = torch.onnx.export(module, tuple(sample), dynamo=True, opset_version=opset, input_names=names,
                                    output_names=["logits", "probabilities"], dynamic_shapes=dynamic_shapes)
    program.save(path, external_data=False)
    # every node carries the Python stack trace it was traced from: absolute local paths and export.py line numbers,
    # which would make the graph's bytes (and its SHA-256) depend on where and from which revision it was exported
    import onnx
    graph = onnx.load(path)
    for node in [*graph.graph.node, *(n for f in graph.functions for n in f.node)]:
        keep = [p for p in node.metadata_props if p.key != "pkg.torch.onnx.stack_trace"]
        del node.metadata_props[:]
        node.metadata_props.extend(keep)
    onnx.save(graph, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="cua-ai/cua-s1-forms")
    ap.add_argument("--revision", help="commit to pin; default: current main")
    ap.add_argument("--name", default="cua-s1-forms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default="../fixtures/cua-s1-forms.json")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--opset", type=int, default=18)
    a = ap.parse_args()

    rev = HfApi().model_info(a.repo, revision=a.revision).sha
    print(f"{a.repo}@{rev}")
    weights = hf_hub_download(a.repo, f"{a.name}.safetensors", revision=rev)
    sidecar = hf_hub_download(a.repo, f"{a.name}.json", revision=rev)
    assert os.path.dirname(weights) == os.path.dirname(sidecar)
    model, collator, config = load_checkpoint(weights, "cpu")          # eval mode, signature checked
    wrapped = Exported(model).eval()

    rows = [r for seed in range(a.episodes) for r in episode_rows(10_000 + seed)]
    sample = collate(collator, rows[:8])

    # everything but the manifest lives under r-<commit>/: republishing never overwrites a file an older manifest
    # names, and the switch to a new checkpoint is the single commit that replaces manifest.json
    rdir = f"r-{rev[:7]}"
    if os.path.isdir(a.out): shutil.rmtree(a.out)
    os.makedirs(os.path.join(a.out, rdir))
    onnx_path = os.path.join(a.out, rdir, "model.onnx")
    from torch.export import Dim
    B, L, N, T = Dim("batch"), Dim("context_len", max=config["context_tokens"]), Dim("options"), Dim("option_len", max=config["option_tokens"])
    export(wrapped, sample, INPUTS, {"context_ids": {0: B, 1: L}, "context_mask": {0: B, 1: L},
                                     "option_ids": {0: B, 1: N, 2: T}, "option_token_mask": {0: B, 1: N, 2: T},
                                     "option_mask": {0: B, 1: N}}, onnx_path, a.opset)
    shared = SharedOptions(model).eval()
    shared_path = os.path.join(a.out, rdir, "model-shared-options.onnx")
    export(shared, collate_shared(collator, rows[:8]), SHARED_INPUTS,
           {"context_ids": {0: B, 1: L}, "context_mask": {0: B, 1: L}, "option_ids": {0: N, 1: T}, "option_token_mask": {0: N, 1: T}},
           shared_path, a.opset)

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst, flips, correct, n, fixtures = 0.0, 0, 0, 0, []
    for i in range(0, len(rows), 16):   # batches of mixed lengths and option counts, as a browser would send
        chunk = rows[i:i + 16]
        tensors = collate(collator, chunk)
        with torch.no_grad():
            _, ref = wrapped(*tensors)
        got = sess.run(["probabilities"], {k: t.numpy() for k, t in zip(INPUTS, tensors)})[0]
        for j, r in enumerate(chunk):
            k = len(r["options"])
            p_ref, p_got = ref[j, :k].numpy(), got[j, :k]
            worst = max(worst, float(np.abs(p_ref - p_got).max()))
            flips += int(p_ref.argmax() != p_got.argmax()); correct += int(p_ref.argmax() == r["label"]); n += 1
            fixtures.append({"context": r["context"], "options": r["options"], "label": r["label"],
                             "action": r["meta"]["action"], "probs": p_ref.tolist()})
    parity = {"decisions": n, "max_abs_dp": worst, "argmax_flips": flips, "top1_vs_labels": correct / n}
    print(json.dumps(parity))

    # the shared graph against the unmodified model's per-row forward, one episode (one option list) per run
    sess = ort.InferenceSession(shared_path, providers=["CPUExecutionProvider"])
    worst, flips, correct, n = 0.0, 0, 0, 0
    for seed in range(a.episodes):
        episode = episode_rows(10_000 + seed)
        with torch.no_grad():
            _, ref = wrapped(*collate(collator, episode))
        got = sess.run(["probabilities"], {k: t.numpy() for k, t in zip(SHARED_INPUTS, collate_shared(collator, episode))})[0]
        worst = max(worst, float(np.abs(ref.numpy() - got).max()))
        flips += int((ref.argmax(-1).numpy() != got.argmax(-1)).sum())
        correct += sum(int(got[j].argmax() == r["label"]) for j, r in enumerate(episode)); n += len(episode)
    shared_parity = {"decisions": n, "max_abs_dp": worst, "argmax_flips": flips, "top1_vs_labels": correct / n}
    print("shared options", json.dumps(shared_parity))

    sha = hashlib.sha256(open(onnx_path, "rb").read()).hexdigest()
    shutil.copy(sidecar, os.path.join(a.out, rdir, "checkpoint.json"))    # upstream config, signature and metadata
    manifest = {
        "name": a.name, "source": f"{a.repo}@{rev}", "model": f"{rdir}/model.onnx", "checkpoint": f"{rdir}/checkpoint.json",
        "sha256": sha, "bytes": os.path.getsize(onnx_path),
        "context_tokens": config["context_tokens"], "option_tokens": config["option_tokens"],
        "inputs": INPUTS, "outputs": ["logits", "probabilities"], "opset": a.opset, "parity": parity,
        # one option list for every row: options encoded once. Clients that predate it read only the fields above.
        "shared_options": {
            "model": f"{rdir}/model-shared-options.onnx", "sha256": hashlib.sha256(open(shared_path, "rb").read()).hexdigest(),
            "bytes": os.path.getsize(shared_path), "inputs": SHARED_INPUTS, "parity": shared_parity,
        },
    }
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    os.makedirs(os.path.dirname(a.fixtures), exist_ok=True)
    json.dump({"source": manifest["source"], "fixtures": fixtures}, open(a.fixtures, "w"))
    print(f"{onnx_path}: {manifest['bytes'] / 1e6:.2f} MB, {shared_path}: {manifest['shared_options']['bytes'] / 1e6:.2f} MB; "
          f"{len(fixtures)} fixtures -> {a.fixtures}")


if __name__ == "__main__":
    main()
