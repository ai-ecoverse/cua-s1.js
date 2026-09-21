"""Publish the ONNX export to a Hugging Face repo (default ai-ecoverse/cua-s1.js).

    HF_TOKEN=... uv run python upload_hf.py --models ../public/models

For each model directory: the revision's files (r-<commit>/) first, then manifest.json alone as the switch, then
whatever the new manifest no longer names. A model card is written from the manifests."""
import argparse, json, os
from huggingface_hub import HfApi

CARD = """---
license: mit
library_name: onnxruntime-web
pipeline_tag: other
base_model:
{bases}
tags: [cua-s1, computer-use, form-filling, system-one, jev, onnx, onnxruntime-web]
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

Each folder has a `manifest.json` naming its source commit, the ONNX graph's SHA-256 (checked by the loader) and
the parity measured at export time. The graph and Cua's original JSON sidecar (`checkpoint.json`: architecture,
tensor signature, training metadata) live under `r-<commit>/`, so publishing a new checkpoint never changes a file
an older manifest points at.

## Conversion

The checkpoint is loaded with `cua_s1.model.load_checkpoint`, which validates the SHA-256 tensor signature. It is
exported unmodified with the `torch.export`-based ONNX exporter (opset 18, dynamic batch, context, option and
option-token axes). It takes the byte tensors `cua_s1`'s `ByteCollator` produces and returns per-option logits and
probabilities. onnxruntime matches PyTorch to within the max |Δp| in the table, on 1,048 decisions from Cua's own
synthetic episode generator, with no argmax flips.

cua-s1 is a research checkpoint trained on synthetic forms. Read Cua's
[model card](https://huggingface.co/{base}) and
[SECURITY.md](https://github.com/trycua/cua/blob/main/libs/cua-s1/SECURITY.md) before relying on it.
"""


def published(name, m):
    return [f"{name}/{p}" for p in ["manifest.json", m["model"], m["checkpoint"]]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../public/models")
    ap.add_argument("--repo", default="ai-ecoverse/cua-s1.js")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = sorted(d for d in os.listdir(a.models) if os.path.exists(f"{a.models}/{d}/manifest.json"))
    manifests = {n: json.load(open(f"{a.models}/{n}/manifest.json")) for n in names}
    rows = ["| Folder | Source checkpoint | ONNX | Max \\|Δp\\| vs PyTorch |", "|---|---|---|---|"]
    for n, m in manifests.items():
        rows.append(f"| `{n}` | [`{m['source'].split('@')[0]}`](https://huggingface.co/{m['source'].split('@')[0]}) @ `{m['source'].split('@')[1][:7]}` "
                    f"| {m['bytes'] / 1e6:.1f} MB | {m['parity']['max_abs_dp']:.1e} over {m['parity']['decisions']} decisions |")
    bases = sorted({m["source"].split("@")[0] for m in manifests.values()})
    card = CARD.format(repo=a.repo, table="\n".join(rows), bases="\n".join(f"- {b}" for b in bases), base=bases[0])
    if a.dry_run:
        print(card); return
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    remote = set(api.list_repo_files(a.repo))
    for n in names:
        files = published(n, manifests[n])
        payload = [f for f in files if not f.endswith("/manifest.json")]
        api.upload_folder(repo_id=a.repo, folder_path=a.models, allow_patterns=payload, commit_message=f"{n}: {manifests[n]['source']} files")
        api.upload_file(path_or_fileobj=f"{a.models}/{n}/manifest.json", path_in_repo=f"{n}/manifest.json", repo_id=a.repo,
                        commit_message=f"{n}: {manifests[n]['source']}")
        stale = sorted(f for f in remote if f.startswith(f"{n}/") and f not in files)
        if stale:
            api.delete_files(repo_id=a.repo, delete_patterns=stale, commit_message=f"{n}: drop {len(stale)} superseded files")
        print(f"{n}: {len(files)} files, {len(stale)} superseded removed")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo, commit_message="Model card")
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
