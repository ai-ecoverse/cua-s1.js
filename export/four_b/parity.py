"""The browser algorithm in Python against onnxruntime, checked against FourBModel's probabilities (fixtures.py).

    uv run python -m four_b.parity --model build/cua-s1-4b-0.1/web-q8f32/model.onnx \
        --head build/cua-s1-4b-0.1/head.safetensors --fixtures ../fixtures/cua-s1-4b-0.1.json

One pass over the whole prompt with empty caches; the final position's hidden state (after the final norm) times the
option letters' output rows, softmaxed over the task's letters, is FourBModel.forward's readout."""
import argparse, json
import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file


class OrtFourB:
    def __init__(self, model_path, head_path, providers=("CPUExecutionProvider",)):
        self.sess = ort.InferenceSession(model_path, providers=list(providers))
        self.dtype = np.float16 if any(i.type == "tensor(float16)" for i in self.sess.get_inputs()) else np.float32
        self.head = load_file(head_path)["weight"].astype(np.float64)          # [26, hidden]

    def feeds(self, ids):
        S = len(ids)
        f = {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, S), np.int64),
             "position_ids": np.broadcast_to(np.arange(S, dtype=np.int64), (3, 1, S)).copy()}   # text: 3 equal mRoPE rows
        for i in self.sess.get_inputs():
            if i.name.startswith("past_key_values."): f[i.name] = np.zeros((1, i.shape[1], 0, 256), self.dtype)
            elif i.name.startswith("past."): f[i.name] = np.zeros([1] + i.shape[1:], self.dtype)
        return f

    def probs(self, ids, n):
        (h,) = self.sess.run(["hidden_states"], self.feeds(ids))
        z = self.head[:n] @ h[0, -1].astype(np.float64)
        e = np.exp(z - z.max())
        return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--fixtures", required=True)
    a = ap.parse_args()
    fx = json.load(open(a.fixtures))
    rt = OrtFourB(a.model, a.head)
    worst, flips, hits, n = 0.0, 0, 0, 0
    for f in fx["fixtures"]:
        ref = np.array(f["probs"])
        got = rt.probs(f["input_ids"], len(ref))
        worst = max(worst, float(np.abs(got - ref).max())); flips += int(got.argmax() != ref.argmax())
        hits += int("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[got.argmax()] in f["gold"]); n += 1
    print(json.dumps({"model": a.model, "tasks": n, "max_abs_dp": round(worst, 6), "argmax_flips": flips, "top1_in_gold": hits}))


if __name__ == "__main__":
    main()
