"""Reference fixtures for cua-s1-4b: tasks from cua-bench-s1's own generator, scored by upstream's FourBModel.

    uv run python -m four_b.fixtures --adapter cua-ai/cua-s1-4b-0.1@<sha> --out ../fixtures/cua-s1-4b-0.1.json

Every task records what the browser has to reproduce: the options, the rendered chat text and its token ids (for the
prompt and tokenizer ports), the letters' token ids, and FourBModel's probabilities in fp32 (for the graph). The gold
letters come from the task's `expected` map, as in train_4b.gold_option_letters: a task can have several."""
import argparse, json, tempfile
from pathlib import Path
from . import pin, snapshot


def tasks(per_app: int, seed: int):
    from cua_bench_s1.datagen import specs
    from cua_bench_s1.datagen.generator import generate_task
    apps = next(v for v in vars(specs).values() if isinstance(v, (list, tuple)) and v and hasattr(v[0], "app_id"))
    out = Path(tempfile.mkdtemp())
    for i, app in enumerate(apps):
        for k in range(per_app):
            # plain, hard-negative and hard-distractor screens in turn, as the generator's datasets mix them
            t = generate_task(app, seed + 1000 * i + k, ("text",), out, hard_negative=k % 3 == 1, hard_distractor=k % 3 == 2)
            if len(t.options) <= 26: yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="cua-ai/cua-s1-4b-0.1")
    ap.add_argument("--per-app", type=int, default=3)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from cua_s1.four_b import FourBModel, Option, assign_letters, build_prompt
    run = pin(a.adapter)
    adir = snapshot(run)
    cfg = json.load(open(f"{adir}/text/adapter_config.json"))
    base = pin(cfg["base_model_name_or_path"])
    bdir = snapshot(base)
    tok = AutoTokenizer.from_pretrained(bdir)
    model = FourBModel(base_model=bdir, lora_adapter_path=adir, device=a.device, dtype="float32", modality="text")
    model.load()
    fixtures = []
    for t in tasks(a.per_app, a.seed):
        options = [Option(o.element_id, o.role, o.label, o.action, o.entity_id) for o in t.options]
        assignment = assign_letters(options)
        chat = tok.apply_chat_template(build_prompt(assignment, app=t.app, task_family=t.family, ax_tree=t.ax_tree),
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(chat).input_ids
        with torch.no_grad():
            probs = [p.probability for p in model.forward(options, app=t.app, task_family=t.family, ax_tree=t.ax_tree)]
        gold = [l for l, o in zip(assignment.letters, options) if t.expected.get(o.element_id) == o.action]
        fixtures.append({"id": t.id, "family": t.family, "app": t.app, "ax_tree": t.ax_tree,
                         "options": [o.__dict__ for o in options], "gold": gold,
                         "chat": chat, "input_ids": ids, "probs": probs})
        top = max(range(len(probs)), key=probs.__getitem__)
        print(f"{t.id}: {len(ids)} tokens, {len(options)} options, top {assignment.letters[top]} "
              f"p={probs[top]:.3f}, gold {''.join(gold)}", flush=True)
    hit = sum(f["gold"] and "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[max(range(len(f["probs"])), key=f["probs"].__getitem__)] in f["gold"]
              for f in fixtures)
    json.dump({"run": run, "base": base, "letter_ids": [tok.encode(l, add_special_tokens=False)[0] for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
               "fixtures": fixtures}, open(a.out, "w"))
    print(f"{len(fixtures)} tasks, top-1 in gold {hit}/{sum(bool(f['gold']) for f in fixtures)} -> {a.out}")


if __name__ == "__main__":
    main()
