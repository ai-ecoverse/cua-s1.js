"""Publish the ONNX export to a Hugging Face repo (default ai-ecoverse/cua-s1.js).

    HF_TOKEN=... uv run python upload_hf.py --models ../public/models

For each model directory: the revision's files (r-<commit>/) and manifest.json in one commit, so a client never reads
a manifest whose graphs are not there yet or were replaced, then whatever the new manifest no longer names. A model
card is written from the manifests."""
import argparse, json, os
from huggingface_hub import HfApi

CARD = """---
license: {license}
library_name: onnxruntime-web
pipeline_tag: other
base_model:
{bases}
tags: [cua-s1, computer-use, form-filling, system-one, jev, onnx, onnxruntime-web{tags}]
---

# cua-s1.js weights

Browser-ready ONNX exports of [Cua's cua-s1](https://github.com/trycua/cua/tree/main/libs/cua-s1) form-filling
decision models, for [@ai-ecoverse/cua-s1.js](https://github.com/ai-ecoverse/cua-s1.js) and onnxruntime-web. The
model, its training data and the planning rules are Cua's work (MIT); this repo only holds the converted
checkpoint.

```js
import * as ort from "onnxruntime-web/wasm";
import {{ loadCuaS1, extractEntities }} from "@ai-ecoverse/cua-s1.js";

const model = await loadCuaS1("https://huggingface.co/{repo}/resolve/main/cua-s1-forms", {{ ort }});
const plan = await model.plan("Northwind Clinic - New Patient Registration",
  [{{ role: "Edit", label: "Phone number", value: "", token: "phone" }}],
  extractEntities("Tel: (503) 555-0142\\nWork phone: (503) 555-0110"));
```

## Contents

{table}

Each folder has a `manifest.json` naming its source commit, the ONNX graphs' SHA-256s (checked by the loader) and
the parity measured at export time. The graphs and Cua's original JSON sidecar (`checkpoint.json`: architecture,
tensor signature, training metadata) live under `r-<commit>/`, so publishing a new checkpoint never changes a file
an older manifest points at.

## Conversion

The checkpoint is loaded with `cua_s1.model.load_checkpoint`, which validates the SHA-256 tensor signature. It is
exported unmodified with the `torch.export`-based ONNX exporter (opset 18, dynamic batch, context, option and
option-token axes). It takes the byte tensors `cua_s1`'s `ByteCollator` produces and returns per-option logits and
probabilities. onnxruntime matches PyTorch to within the max |Δp| in the table, on 1,048 decisions from Cua's own
synthetic episode generator, with no argmax flips.

`model-shared-options.onnx` runs the same modules for a batch whose rows all share one option list, which is every
form plan: the options are encoded once instead of once per element. Its inputs drop the batch axis from the option
tensors. It gives the same probabilities as `model.onnx` and is checked against the unmodified PyTorch forward on the
same episodes.

cua-s1 is a research checkpoint trained on synthetic forms. Read Cua's
[model card](https://huggingface.co/{base}) and
[SECURITY.md](https://github.com/trycua/cua/blob/main/libs/cua-s1/SECURITY.md) before relying on it.
{four_b}"""

FOUR_B = """
## cua-s1-4b

`{name}/` is [`{adapter}`](https://huggingface.co/{adapter})'s text adapter (a rank-16 LoRA) merged in fp32 into
[`{base}`](https://huggingface.co/{base}) and exported with the onnxruntime-genai model builder: int8 weights, fp32
activations, WebGPU. The graph returns hidden states; Qwen3.5-4B ties its output layer to the embeddings, so the
option letters' logits are the final hidden state times `head.safetensors` (the 26 letter rows, fp32). Weights are
split into files of at most 32 MB.

```js
import * as ort from "onnxruntime-web/webgpu";
import {{ loadCuaS1FourB, renderAxTree }} from "@ai-ecoverse/cua-s1.js/4b";

const model = await loadCuaS1FourB("https://huggingface.co/{repo}/resolve/main/{name}", {{ ort }});
const r = await model.score(options, {{ app, taskFamily: "form_filling", axTree: renderAxTree(title, rows, entities) }});
```

Against Cua's own `cua_s1.four_b.FourBModel` (fp32 PyTorch) on {tasks} tasks from cua-bench-s1's generator: max
|Δp| {dp:.4f}, {flips} argmax flips (near-ties). The fp32 export matches to 7.5e-5.

Licensing: Qwen3.5-4B is Apache-2.0. Cua publishes the cua-s1-4b-0.1 adapter without a license file, and its
model card notes that official checkpoints may carry their own terms. Check with Cua before using this folder
beyond research and evaluation.
"""


def published(name, m):
    if "variants" in m:   # cua-s1-4b: tokenizer, head, and each variant's graph and weight shards
        files = [*m["files"].values(), *(p for v in m["variants"].values() for p in [v["model"], *v["data"]])]
    else:
        files = [m["model"], *([m["shared_options"]["model"]] if "shared_options" in m else []), m["checkpoint"]]
    return [f"{name}/{p}" for p in ["manifest.json", *files]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../public/models")
    ap.add_argument("--repo", default="ai-ecoverse/cua-s1.js")
    ap.add_argument("--only", action="append", help="publish just these model folders (the card still lists all)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = sorted(d for d in os.listdir(a.models) if os.path.exists(f"{a.models}/{d}/manifest.json"))
    manifests = {n: json.load(open(f"{a.models}/{n}/manifest.json")) for n in names}
    rows = ["| Folder | Source checkpoint | ONNX | Max \\|Δp\\| vs PyTorch |", "|---|---|---|---|"]
    four_b = ""
    for n, m in manifests.items():
        src = m.get("source") or m["run"]
        repo, rev = src.split("@")
        if "variants" in m:
            v = next(iter(m["variants"].values()))
            size, parity = f"{v['bytes'] / 1e9:.1f} GB", f"{v['parity']['max_abs_dp']:.1e} over {v['parity']['tasks']} tasks"
            four_b += FOUR_B.format(name=n, adapter=repo, base=m["base"].split("@")[0], repo=a.repo, tasks=v["parity"]["tasks"],
                                    dp=v["parity"]["max_abs_dp"], flips=v["parity"]["argmax_flips"])
        else:
            size, parity = f"{m['bytes'] / 1e6:.1f} MB", f"{m['parity']['max_abs_dp']:.1e} over {m['parity']['decisions']} decisions"
        rows.append(f"| `{n}` | [`{repo}`](https://huggingface.co/{repo}) @ `{rev[:7]}` | {size} | {parity} |")
    bases = sorted({(m.get("source") or m["run"]).split("@")[0] for m in manifests.values()}
                   | {m["base"].split("@")[0] for m in manifests.values() if "base" in m})
    card = CARD.format(repo=a.repo, table="\n".join(rows), bases="\n".join(f"- {b}" for b in bases),
                       base=next(b for b in bases if b.endswith("forms")) if any(b.endswith("forms") for b in bases) else bases[0],
                       license="other" if four_b else "mit", tags=", qwen3.5, lora" if four_b else "", four_b=four_b)
    if a.dry_run:
        print(card); return
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    remote = set(api.list_repo_files(a.repo))
    for n in names:
        if a.only and n not in a.only: continue
        files = published(n, manifests[n])
        api.upload_folder(repo_id=a.repo, folder_path=a.models, allow_patterns=files,
                          commit_message=f"{n}: {manifests[n].get('source') or manifests[n]['run']}")
        stale = sorted(f for f in remote if f.startswith(f"{n}/") and f not in files)
        if stale:
            api.delete_files(repo_id=a.repo, delete_patterns=stale, commit_message=f"{n}: drop {len(stale)} superseded files")
        print(f"{n}: {len(files)} files, {len(stale)} superseded removed")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo, commit_message="Model card")
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
