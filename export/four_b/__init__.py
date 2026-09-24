"""Export cua-s1-4b (a LoRA on Qwen/Qwen3.5-4B) for the browser: merge the text adapter, build the ONNX graph with the
onnxruntime-genai model builder (hidden states, no LM head), keep the option letters' output rows as the head, and
measure the result against upstream's own cua_s1.four_b.FourBModel."""
import os


def pin(repo: str) -> str:
    """repo[@rev] -> repo@<full commit sha>. Hub ids can be republished in place; every step works on the pinned form."""
    if os.path.isdir(repo): return repo
    name, _, rev = repo.partition("@")
    if len(rev) == 40: return repo
    from huggingface_hub import HfApi
    return f"{name}@{HfApi().model_info(name, revision=rev or None).sha}"


def snapshot(pinned: str) -> str:
    if os.path.isdir(pinned): return pinned   # a local adapter or base, as pin() passes it through
    from huggingface_hub import snapshot_download
    name, _, rev = pinned.partition("@")
    return snapshot_download(name, revision=rev)
