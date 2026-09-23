"""Assemble the cua-s1-4b web bundle: manifest.json, tokenizer, letter head and one directory per ONNX variant.

    uv run python -m four_b.package --build build/cua-s1-4b-0.1 --out ../public/models/cua-s1-4b-0.1 \
        --variant q8f32=web-q8f32 --fixtures ../fixtures/cua-s1-4b-0.1.json

Large files are hard-linked, not copied. Everything but the manifest lives under r-<adapter commit>/, so publishing
a new checkpoint never overwrites a file an older manifest points at. With --fixtures, each variant's parity against
FourBModel is measured (CPU EP) and recorded."""
import argparse, json, os, shutil
import onnx
from .parity import OrtFourB

ONNX_TYPES = {onnx.TensorProto.FLOAT16: "float16", onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.INT64: "int64"}


def io_info(model_path):
    m = onnx.load(model_path, load_external_data=False)
    def info(v, inp):
        shape = [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
        out = {"name": v.name, "type": ONNX_TYPES[v.type.tensor_type.elem_type], "shape": shape}
        if inp and v.name.startswith("past_key_values."):
            out["empty"] = [1, shape[1], 0, 256]          # kv_cache_dim = head_dim of the full-attention layers
        elif inp and v.name.startswith("past."):
            out["empty"] = [1] + shape[1:]
        return out
    return [info(v, True) for v in m.graph.input], [info(v, False) for v in m.graph.output]


def link(src, dst):
    if os.path.exists(dst): os.remove(dst)
    try: os.link(src, dst)
    except OSError: shutil.copy(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="four_b.merge output dir (cua4b.json, head.safetensors, tokenizer/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", action="append", required=True, help="name=dir, e.g. q8f32=web-q8f32")
    ap.add_argument("--fixtures")
    a = ap.parse_args()
    meta = json.load(open(f"{a.build}/cua4b.json"))
    r = f"r-{meta['run'].partition('@')[2][:7] or 'local'}"
    os.makedirs(f"{a.out}/{r}", exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json"): link(f"{a.build}/tokenizer/{f}", f"{a.out}/{r}/{f}")
    link(f"{a.build}/head.safetensors", f"{a.out}/{r}/head.safetensors")
    fixtures = None
    if a.fixtures:
        fx = json.load(open(a.fixtures))
        if fx["run"] != meta["run"]:   # parity is only meaningful against the checkpoint the weights came from
            raise SystemExit(f"{a.fixtures} is from {fx['run']}, but {a.build} was exported from {meta['run']}")
        fixtures = fx["fixtures"]
    shared = {f"{r}/{f}": os.path.getsize(f"{a.out}/{r}/{f}") for f in ("head.safetensors", "tokenizer.json", "tokenizer_config.json")}
    variants = {}
    for spec in a.variant:
        name, d = spec.split("=", 1)
        vdir = f"{r}/{name}"
        os.makedirs(f"{a.out}/{vdir}", exist_ok=True)
        link(f"{a.build}/{d}/model.onnx", f"{a.out}/{vdir}/model.onnx")
        data = sorted((f for f in os.listdir(f"{a.build}/{d}") if f.startswith("model.onnx.data")), key=lambda f: (len(f), f))
        for f in data: link(f"{a.build}/{d}/{f}", f"{a.out}/{vdir}/{f}")
        inputs, outputs = io_info(f"{a.out}/{vdir}/model.onnx")
        v = {"model": f"{vdir}/model.onnx", "data": [f"{vdir}/{f}" for f in data],
             "bytes": sum(os.path.getsize(f"{a.out}/{vdir}/{f}") for f in ["model.onnx", *data]),
             "io_dtype": next(o["type"] for o in outputs if o["name"] == "hidden_states"),
             "sizes": {**{f"{vdir}/{f}": os.path.getsize(f"{a.out}/{vdir}/{f}") for f in ["model.onnx", *data]}, **shared},
             "inputs": inputs, "outputs": outputs}
        if fixtures:
            import numpy as np
            rt = OrtFourB(f"{a.out}/{vdir}/model.onnx", f"{a.out}/{r}/head.safetensors")
            worst, flips, hits = 0.0, 0, 0
            for f in fixtures:
                ref = np.array(f["probs"]); got = rt.probs(f["input_ids"], len(ref))
                worst = max(worst, float(np.abs(got - ref).max())); flips += int(got.argmax() != ref.argmax())
                hits += int("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[got.argmax()] in f["gold"])
            v["parity"] = {"max_abs_dp": round(worst, 6), "argmax_flips": flips, "tasks": len(fixtures), "top1_in_gold": hits}
            print(name, v["parity"])
        variants[name] = v
    keep = {"manifest.json", *shared}
    for v in variants.values(): keep |= {v["model"], *v["data"]}
    for root, _, fs in os.walk(a.out, topdown=False):   # drop files from earlier packagings: the directory is published as is
        for f in fs:
            rel = os.path.relpath(os.path.join(root, f), a.out)
            if rel not in keep: os.remove(os.path.join(root, f)); print("removed stale", rel)
        if root != a.out and not os.listdir(root): os.rmdir(root)
    manifest = {"name": os.path.basename(os.path.normpath(a.out)), **{k: meta[k] for k in ("run", "modality", "base", "hidden_size", "letters", "letter_ids")},
                "files": {"head": f"{r}/head.safetensors", "tokenizer": f"{r}/tokenizer.json", "tokenizer_config": f"{r}/tokenizer_config.json"},
                "variants": variants}
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=2)
    print(f"{a.out}/manifest.json: {', '.join(f'{k} {v['bytes'] / 1e6:.0f} MB' for k, v in variants.items())}")


if __name__ == "__main__":
    main()
