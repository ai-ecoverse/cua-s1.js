"""The browser algorithm in Python against onnxruntime, checked against FourBModel's probabilities (fixtures.py).

    uv run python -m four_b.parity --model build/cua-s1-4b-0.2/web-q8f32/model.onnx \
        --head build/cua-s1-4b-0.2/head.safetensors --fixtures ../fixtures/cua-s1-4b-0.2.json

One pass over the whole prompt with empty caches; the final position's hidden state (after the final norm) times the
option letters' output rows, softmaxed over the task's letters, is FourBModel.forward's readout.

Multimodal (--vision, fixtures from `fixtures.py --modality multimodal`): Qwen's processor makes the patches, the
vision graph turns them into one row per 2x2 block, the decoder splices those rows in at the <|image_pad|> tokens,
and positions follow Qwen3_5Model.get_rope_index: text counts up on all three mRoPE axes; the image's tokens get
(start, start + row, start + column) over the merged grid, and text after it resumes at start + max(rows, columns)."""
import argparse, json
import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file
from .fixtures import summary


class OrtFourB:
    def __init__(self, model_path, head_path, providers=("CPUExecutionProvider",)):
        self.sess = ort.InferenceSession(model_path, providers=list(providers))
        self.dtype = np.float16 if any(i.type == "tensor(float16)" for i in self.sess.get_inputs()) else np.float32
        self.head = load_file(head_path)["weight"].astype(np.float64)          # [26, hidden]

    def feeds(self, ids, pos=None, image_embeds=None):
        S = len(ids)
        f = {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, S), np.int64),
             "position_ids": pos if pos is not None else np.broadcast_to(np.arange(S, dtype=np.int64), (3, 1, S)).copy()}   # text: 3 equal mRoPE rows
        if any(i.name == "image_embeds" for i in self.sess.get_inputs()):
            f["image_embeds"] = image_embeds if image_embeds is not None else np.zeros((1, self.head.shape[1]), self.dtype)
        for i in self.sess.get_inputs():
            if i.name.startswith("past_key_values."): f[i.name] = np.zeros((1, i.shape[1], 0, 256), self.dtype)
            elif i.name.startswith("past."): f[i.name] = np.zeros([1] + i.shape[1:], self.dtype)
        return f

    def probs(self, ids, n, pos=None, image_embeds=None):
        (h,) = self.sess.run(["hidden_states"], self.feeds(ids, pos, image_embeds))
        z = self.head[:n] @ h[0, -1].astype(np.float64)
        e = np.exp(z - z.max())
        return e / e.sum()


IMAGE_PAD, MERGE = 248056, 2


def rope_positions(ids, grid_h, grid_w, image_token=IMAGE_PAD, merge=MERGE):
    """[3, 1, S] mRoPE positions for a prompt with one image (Qwen3_5Model.get_rope_index)."""
    pos, cur, i, S = [], 0, 0, len(ids)
    while i < S:
        if ids[i] != image_token:
            pos.append((cur, cur, cur)); cur += 1; i += 1; continue
        gh, gw = grid_h // merge, grid_w // merge
        for r in range(gh):
            for c in range(gw): pos.append((cur, cur + r, cur + c))
        i += gh * gw; cur += max(gh, gw)
    return np.array(pos, np.int64).T.reshape(3, 1, S)


class OrtVision:
    def __init__(self, model_path, merged):
        from .vision import load_visual
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.v = load_visual(merged)

    def embeds(self, pixel_values, grid_h, grid_w):
        from .vision import vision_inputs, INPUTS
        ins = [pixel_values, *(t.numpy() for t in vision_inputs(self.v, grid_h, grid_w))]
        return self.sess.run(["image_embeds"], dict(zip(INPUTS, [np.asarray(x, np.float32) if x.dtype != np.int64 else x for x in ins])))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--vision", help="vision graph (multimodal fixtures)")
    ap.add_argument("--merged", help="merged checkpoint dir: vision config, position table and the processor")
    a = ap.parse_args()
    fx = json.load(open(a.fixtures))
    rt = OrtFourB(a.model, a.head)
    if a.vision:
        import os
        from PIL import Image
        from transformers import AutoImageProcessor
        vis, proc = OrtVision(a.vision, a.merged), AutoImageProcessor.from_pretrained(a.merged)
    worst, flips, n, got_all = 0.0, 0, 0, []
    for f in fx["fixtures"]:
        ref = np.array(f["probs"])
        if a.vision:
            img = Image.open(os.path.join(os.path.dirname(a.fixtures), f["screenshot"])).convert("RGB")
            px = proc(images=[img], return_tensors="np")
            _, gh, gw = px["image_grid_thw"][0].tolist()
            got = rt.probs(f["input_ids"], len(ref), rope_positions(f["input_ids"], gh, gw), vis.embeds(px["pixel_values"], gh, gw))
        else:
            got = rt.probs(f["input_ids"], len(ref))
        worst = max(worst, float(np.abs(got - ref).max())); flips += int(got.argmax() != ref.argmax())
        got_all.append(got.tolist()); n += 1
    print(json.dumps({"model": a.model, "tasks": n, "max_abs_dp": round(worst, 6), "argmax_flips": flips, "quality": summary(fx["fixtures"], got_all)}))


if __name__ == "__main__":
    main()
