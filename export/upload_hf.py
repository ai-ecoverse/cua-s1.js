"""Publish the ONNX export to a Hugging Face repo (default ai-ecoverse/cua-s1.js).

    HF_TOKEN=... uv run python upload_hf.py --models ../public/models

For each model directory: the revision's files (r-<commit>/) and manifest.json in one commit, so a client never reads
a manifest whose graphs are not there yet or were replaced, then whatever the new manifest no longer names. A model
card is written from the manifests."""
import argparse, json, os
from huggingface_hub import HfApi

CARD = """---
license: {license}{license_extra}
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

`{name}/` is [`{adapter}`](https://huggingface.co/{adapter})'s {modality} adapter (a rank-16 LoRA) merged in fp32 into
[`{base}`](https://huggingface.co/{base}) and exported with the onnxruntime-genai model builder: int8 weights, fp32
activations, WebGPU. The graph returns hidden states; Qwen3.5-4B ties its output layer to the embeddings, so the
option letters' logits are the final hidden state times `head.safetensors` (the 26 letter rows, fp32). Weights are
split into files of at most 32 MB.{vision}

```js
import * as ort from "onnxruntime-web/webgpu";
import {{ elementDecisions, loadCuaS1FourB }} from "@ai-ecoverse/cua-s1.js/4b";

const model = await loadCuaS1FourB("https://huggingface.co/{repo}/resolve/main/{name}", {{ ort }});
const r = await model.score(options, {{ app, taskFamily, goal, {input} }});
elementDecisions(r.options);   // per element, its likeliest action
```

Against Cua's own `cua_s1.four_b.FourBModel` (fp32 PyTorch) on {tasks} real Word, Excel and PowerPoint steps from
[GUI-360](https://huggingface.co/datasets/vyokky/GUI-360)'s test split: max |Δp| {dp:.4f}, {flips} argmax flips. Every
element's likeliest action is the expected one on {quality} of them with this export, as cua-bench-s1 scores it.

Licensing: the adapter and Qwen3.5-4B are both Apache-2.0.
"""


def published(name, m):
    if "variants" in m:   # cua-s1-4b: tokenizer, head, and each variant's graph and weight shards
        files = [*m["files"].values(), *(p for v in [*m["variants"].values(), *([m["vision"]] if "vision" in m else [])] for p in [v["model"], *v["data"]])]
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
            vision = "" if "vision" not in m else (
                f"\n\nThe screenshot adapter also adapts the vision tower, so `vision/` is Qwen3.5's ViT and patch merger from the same "
                f"merge ({m['vision']['bytes'] / 1e6:.0f} MB, fp16 weights, fp32 compute). Its size-dependent inputs (position-table taps, 2D "
                f"rotary angles) are computed by the caller, and the decoder takes its output as `image_embeds` at the "
                f"`<|image_pad|>` tokens.")
            four_b += FOUR_B.format(name=n, adapter=repo, base=m["base"].split("@")[0], repo=a.repo, tasks=v["parity"]["tasks"],
                                    dp=v["parity"]["max_abs_dp"], flips=v["parity"]["argmax_flips"],
                                    quality=v["parity"]["quality"].split(",")[0].removeprefix("task accuracy "),
                                    input="screenshot: imageData" if "vision" in m else "axTree",
                                    modality="screenshot (multimodal)" if "vision" in m else "text", vision=vision)
        else:
            size, parity = f"{m['bytes'] / 1e6:.1f} MB", f"{m['parity']['max_abs_dp']:.1e} over {m['parity']['decisions']} decisions"
        rows.append(f"| `{n}` | [`{repo}`](https://huggingface.co/{repo}) @ `{rev[:7]}` | {size} | {parity} |")
    bases = sorted({(m.get("source") or m["run"]).split("@")[0] for m in manifests.values()}
                   | {m["base"].split("@")[0] for m in manifests.values() if "base" in m})
    card = CARD.format(repo=a.repo, table="\n".join(rows), bases="\n".join(f"- {b}" for b in bases),
                       base=next(b for b in bases if b.endswith("forms")) if any(b.endswith("forms") for b in bases) else bases[0],
                       license="other" if four_b else "mit",
                       license_extra="\nlicense_name: mit-and-apache-2.0" if four_b else "", tags=", qwen3.5, lora" if four_b else "", four_b=four_b)
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
