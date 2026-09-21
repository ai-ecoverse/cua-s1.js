"""Export a cua-s1 checkpoint to ONNX for the browser and verify it against PyTorch.

    uv run python export.py --out ../public/models/cua-s1-forms

1. Pin the Hugging Face repo to a commit (checkpoints can be republished under the same id) and download the
   safetensors + JSON sidecar at that commit.
2. Load it with cua_s1's own loader, which validates the format and the SHA-256 tensor signature.
3. Export the unmodified model to ONNX with dynamic batch, context, option and option-token axes. The graph takes
   the tensors ByteCollator produces and returns per-option logits and probabilities.
4. Score synthetic episodes (cua_s1.synth, the training generator) with both PyTorch and onnxruntime, and write the
   PyTorch outputs as parity fixtures for the JS runtime.
"""
import argparse, hashlib, json, os, shutil
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from cua_s1.model import load_checkpoint
from cua_s1.synth import episode_rows

INPUTS = ["context_ids", "context_mask", "option_ids", "option_token_mask", "option_mask"]


class Exported(torch.nn.Module):
    """TinyTransformerScorer.forward takes a dict; ONNX wants positional tensors."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context_ids, context_mask, option_ids, option_token_mask, option_mask):
        logits = self.model({"context_ids": context_ids, "context_mask": context_mask, "option_ids": option_ids,
                             "option_token_mask": option_token_mask, "option_mask": option_mask})
        return logits, logits.softmax(-1)


def collate(collator, rows):
    from cua_s1.model import ChoiceExample
    batch = collator([ChoiceExample(context=r["context"], options=r["options"], label=r.get("label", 0)) for r in rows])
    return [batch[k] for k in INPUTS]


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

    os.makedirs(a.out, exist_ok=True)
    onnx_path = os.path.join(a.out, "model.onnx")
    # torch.export-based exporter: the legacy tracer bakes the sample's sequence length into nn.MultiheadAttention's
    # reshapes. Grad enabled + parameters requiring grad keeps nn.TransformerEncoderLayer off its fused inference
    # fast path (torch._transformer_encoder_layer_fwd); eval() still disables dropout.
    from torch.export import Dim
    B, L, N, T = Dim("batch"), Dim("context_len", max=config["context_tokens"]), Dim("options"), Dim("option_len", max=config["option_tokens"])
    with torch.enable_grad():
        program = torch.onnx.export(
            wrapped, tuple(sample), dynamo=True, opset_version=a.opset,
            input_names=INPUTS, output_names=["logits", "probabilities"],
            dynamic_shapes={"context_ids": {0: B, 1: L}, "context_mask": {0: B, 1: L},
                            "option_ids": {0: B, 1: N, 2: T}, "option_token_mask": {0: B, 1: N, 2: T},
                            "option_mask": {0: B, 1: N}},
        )
    program.save(onnx_path, external_data=False)

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

    sha = hashlib.sha256(open(onnx_path, "rb").read()).hexdigest()
    shutil.copy(sidecar, os.path.join(a.out, "checkpoint.json"))    # upstream config, signature and metadata
    manifest = {
        "name": a.name, "source": f"{a.repo}@{rev}", "model": "model.onnx", "sha256": sha, "bytes": os.path.getsize(onnx_path),
        "context_tokens": config["context_tokens"], "option_tokens": config["option_tokens"],
        "inputs": INPUTS, "outputs": ["logits", "probabilities"], "opset": a.opset, "parity": parity,
    }
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    os.makedirs(os.path.dirname(a.fixtures), exist_ok=True)
    json.dump({"source": manifest["source"], "fixtures": fixtures}, open(a.fixtures, "w"))
    print(f"{onnx_path}: {manifest['bytes'] / 1e6:.2f} MB; {len(fixtures)} fixtures -> {a.fixtures}")


if __name__ == "__main__":
    main()
