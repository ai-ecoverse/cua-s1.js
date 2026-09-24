"""Fold a cua-s1-4b LoRA into Qwen/Qwen3.5-4B in fp32 and write a checkpoint the onnxruntime-genai builder reads.

    uv run python -m four_b.merge --adapter cua-ai/cua-s1-4b-0.2 [--modality multimodal] --out build/cua-s1-4b-0.2

The text and multimodal adapters are trained independently: the multimodal one also adapts the vision tower's MLPs
and its merger (linear_fc1/linear_fc2), so its export takes the vision weights from this merged checkpoint too.

W' = W + (alpha / r) * B @ A for every adapted module, in fp32, which is what peft's merge_and_unload computes. The
builder expects the full Qwen3_5ForConditionalGeneration layout (model.language_model.*, model.visual.*, mtp.*), so
the base checkpoint is copied and only the adapted language-model tensors are replaced.

Qwen3.5-4B ties its output layer to the embeddings and the adapter touches neither, so upstream's readout (the
final-position logits of the option letters) is the final hidden state times the letters' embedding rows. Those rows
are written as head.safetensors in fp32, so the head stays exact whatever the graph's embedding is quantized to."""
import argparse, json, os, shutil
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from . import pin, snapshot

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # cua_s1.four_b.LETTERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="cua-ai/cua-s1-4b-0.2")
    ap.add_argument("--modality", default="text", choices=["text", "multimodal"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = pin(a.adapter)
    adir = f"{snapshot(run)}/{a.modality}"
    cfg = json.load(open(f"{adir}/adapter_config.json"))
    if cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("modules_to_save"):
        raise SystemExit(f"unsupported adapter config: {cfg}")
    base = pin(f"{cfg['base_model_name_or_path']}@{cfg.get('revision') or ''}".rstrip("@"))
    bdir = snapshot(base)
    print(f"adapter {run} ({a.modality}), base {base}")

    idx = json.load(open(f"{bdir}/model.safetensors.index.json"))["weight_map"]
    sd = {}
    for f in sorted(set(idx.values())): sd.update(load_file(f"{bdir}/{f}"))
    scale = cfg["lora_alpha"] / cfg["r"]
    with safe_open(f"{adir}/adapter_model.safetensors", "pt") as f:
        names = sorted({k[: -len(".lora_A.weight")] for k in f.keys() if k.endswith(".lora_A.weight")})
        if 2 * len(names) != len(list(f.keys())):
            raise SystemExit("adapter holds tensors other than lora_A/lora_B pairs")
        for n in names:
            # text (Qwen3_5ForCausalLM): base_model.model.model.layers.N.<m> -> model.language_model.layers.N.<m>
            # multimodal (Qwen3_5ForConditionalGeneration): base_model.model.model.{language_model,visual}.<m> -> as is
            if not n.startswith("base_model.model.model."): raise SystemExit(f"unexpected adapter key {n}")
            key = n[len("base_model.model."):] + ".weight"
            if key.startswith("model.layers."): key = "model.language_model." + key[len("model."):]
            if key not in sd: raise SystemExit(f"{n}: no {key} in the base checkpoint")
            A, B = f.get_tensor(f"{n}.lora_A.weight").float(), f.get_tensor(f"{n}.lora_B.weight").float()
            sd[key] = (sd[key].float() + scale * (B @ A)).contiguous()
    lm = [k for k in sd if k.startswith(("model.language_model.", "model.visual."))]
    for k in lm: sd[k] = sd[k].float().contiguous()             # language model and vision tower in fp32; mtp untouched
    print(f"merged {len(names)} LoRA modules (alpha/r = {scale:g}, {sum('.visual.' in n for n in names)} in the vision tower); "
          f"{len(lm)} language-model and vision tensors in fp32")

    ckpt = os.path.join(a.out, "merged")
    os.makedirs(ckpt, exist_ok=True)
    save_file(sd, f"{ckpt}/model.safetensors", metadata={"format": "pt"})
    for f in os.listdir(bdir):
        if f.endswith((".json", ".txt", ".jinja")) and f != "model.safetensors.index.json":
            shutil.copy(f"{bdir}/{f}", ckpt)
    mcfg = json.load(open(f"{ckpt}/config.json")); mcfg.setdefault("text_config", {})["dtype"] = "float32"
    mcfg.setdefault("eos_token_id", mcfg["text_config"].get("eos_token_id"))   # the builder reads it from the top level
    json.dump(mcfg, open(f"{ckpt}/config.json", "w"), indent=2)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bdir)
    ids = []
    for letter in LETTERS:
        t = tok.encode(letter, add_special_tokens=False)
        if len(t) != 1: raise SystemExit(f"{letter!r} is {len(t)} tokens")   # as FourBModel._letter_token_ids requires
        ids.append(t[0])
    if not mcfg["text_config"].get("tie_word_embeddings", mcfg.get("tie_word_embeddings")):
        raise SystemExit("the base does not tie its output layer to the embeddings: export lm_head rows instead")
    save_file({"weight": sd["model.language_model.embed_tokens.weight"][ids].float().contiguous()}, f"{a.out}/head.safetensors")
    tok.save_pretrained(f"{a.out}/tokenizer")
    meta = {"run": run, "modality": a.modality, "base": base, "hidden_size": mcfg["text_config"]["hidden_size"],
            "letters": LETTERS, "letter_ids": ids}
    json.dump(meta, open(f"{a.out}/cua4b.json", "w"), indent=2)
    print(f"-> {ckpt}; head ({len(ids)} x {meta['hidden_size']}), tokenizer, cua4b.json -> {a.out}")


if __name__ == "__main__":
    main()
