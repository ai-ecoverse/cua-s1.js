"""Reference fixtures for cua-s1-4b: real screens from GUI-360's test split, scored by upstream's FourBModel.

    uv run python -m four_b.fixtures --adapter cua-ai/cua-s1-4b-0.2@<sha> --out ../fixtures/cua-s1-4b-0.2.json

GUI-360 (vyokky/GUI-360, MIT) records Word, Excel and PowerPoint episodes: per step a screenshot, the live UI Automation
tree and the action taken. cua-bench-s1's own converter (datagen.gui360) turns each step into a task, and the task's
goal (the episode's request and the step's subtask, task.goal_text) is stated in the prompt as upstream's eval does.
Most steps are clicks filed under multi_step_submit, so every task of the other families is kept and every
--every'th multi_step_submit task. --source generator takes cua-bench-s1's synthetic screens instead, which is what
cua-s1-4b-0.1 was trained on and 0.2 never saw.

Every task records what the browser has to reproduce: the options, the rendered chat text and its token ids (for the
prompt and tokenizer ports), the letters' token ids, and FourBModel's probabilities in fp32 (for the graph). The gold
letters come from the task's `expected` map, as in train_4b.gold_option_letters: a task can have several.

--modality multimodal keeps each task's screenshot next to the fixtures as <id>.png and records what Qwen's processor
made of it: the patch grid, and statistics of pixel_values for the preprocessing port. GUI-360's screenshots come from
its test/image.tar.gz (--gui360-images, the directory it was extracted to). The generator draws with DejaVu Sans on
Linux; --font-dir supplies it elsewhere (Pillow's bitmap default otherwise)."""
import argparse, glob, json, tempfile
from pathlib import Path
from . import pin, snapshot


def tasks(per_app: int, seed: int, modality: str = "text", out: Path | None = None):
    from cua_bench_s1.datagen import specs
    from cua_bench_s1.datagen.generator import generate_task
    apps = next(v for v in vars(specs).values() if isinstance(v, (list, tuple)) and v and hasattr(v[0], "app_id"))
    out = out or Path(tempfile.mkdtemp())
    for i, app in enumerate(apps):
        for k in range(per_app):
            # plain, hard-negative and hard-distractor screens in turn, as the generator's datasets mix them
            t = generate_task(app, seed + 1000 * i + k, (modality,), out, hard_negative=k % 3 == 1, hard_distractor=k % 3 == 2)
            if len(t.options) <= 26: yield t


GUI360 = "vyokky/GUI-360"


def gui360_tasks(every: int, modality: str = "text", images: Path | None = None, out: Path | None = None, rev: str | None = None):
    from huggingface_hub import snapshot_download
    from cua_bench_s1.datagen.gui360 import convert_episode
    root = snapshot_download(GUI360, repo_type="dataset", revision=rev, allow_patterns=["test/data/*/in_app/success/*.jsonl"])
    out = out or Path(tempfile.mkdtemp())
    k = 0
    for i, f in enumerate(sorted(glob.glob(f"{root}/test/data/*/in_app/success/*.jsonl"))):
        # plain and hard-distractor steps in turn, one kind per episode
        for t in convert_episode(f, images_root=images or Path("/nonexistent"), out_dir=out, modality_available=(modality,),
                                 max_steps_per_episode=100, hard_distractor=i % 2 == 1):
            if t.family == "multi_step_submit":
                k += 1
                if k % every: continue
            if len(t.options) <= 26: yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="cua-ai/cua-s1-4b-0.2")
    ap.add_argument("--source", default="gui360", choices=["gui360", "generator"])
    ap.add_argument("--gui360-rev", default=None, help="dataset revision (default: the current one, recorded in the output)")
    ap.add_argument("--gui360-images", default="build/gui360/image", help="test/image.tar.gz extracted")
    ap.add_argument("--every", type=int, default=10, help="keep every n-th GUI-360 multi_step_submit task")
    ap.add_argument("--per-app", type=int, default=3, help="--source generator: tasks per synthetic app")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--modality", default="text", choices=["text", "multimodal"])
    ap.add_argument("--font-dir", default="build/fonts", help="DejaVuSans.ttf and DejaVuSans-Bold.ttf, as on Linux")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mm = a.modality == "multimodal"
    shots = Path(a.out).with_suffix("") if mm else None
    if mm:
        import os
        from PIL import ImageFont
        from cua_bench_s1.datagen import render
        if os.path.exists(f"{a.font_dir}/DejaVuSans.ttf"):
            render.FONT_TITLE = ImageFont.truetype(f"{a.font_dir}/DejaVuSans-Bold.ttf", 20)
            render.FONT_LABEL = render.FONT_VALUE = ImageFont.truetype(f"{a.font_dir}/DejaVuSans.ttf", 14)
        shots.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoTokenizer
    from cua_s1.four_b import FourBModel, Option, assign_letters, build_prompt
    run = pin(a.adapter)
    adir = snapshot(run)
    cfg = json.load(open(f"{adir}/{a.modality}/adapter_config.json"))
    base = pin(cfg["base_model_name_or_path"])
    bdir = snapshot(base)
    tok = AutoTokenizer.from_pretrained(bdir)
    model = FourBModel(base_model=bdir, lora_adapter_path=adir, device=a.device, dtype="float32", modality=a.modality)
    model.load()
    source = {"generator": None}
    if a.source == "gui360":
        from huggingface_hub import HfApi
        source = {"gui360": f"{GUI360}@{HfApi().dataset_info(GUI360, revision=a.gui360_rev).sha}"}
        it = gui360_tasks(a.every, a.modality, Path(a.gui360_images), shots, source["gui360"].partition("@")[2])
    else:
        it = tasks(a.per_app, a.seed, a.modality, shots)
    fixtures = []
    for t in it:
        options = [Option(o.element_id, o.role, o.label, o.action, o.entity_id) for o in t.options]
        assignment = assign_letters(options)
        shot = str(shots / Path(t.screenshot).name) if mm else None
        goal = t.goal
        messages = build_prompt(assignment, app=t.app, task_family=t.family, ax_tree=t.ax_tree, screenshot=shot, modality=a.modality, goal=goal)
        extra = {}
        if mm:   # as FourBModel.forward: the processor's template, then the processor expands <|image_pad|>
            from PIL import Image
            chat = model._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image = Image.open(shot).convert("RGB")
            inputs = model._processor(text=[chat], images=[image], return_tensors="pt")
            ids = inputs["input_ids"][0].tolist()
            pv = inputs["pixel_values"].double()
            extra = {"screenshot": f"{shots.name}/{Path(shot).name}", "size": list(image.size), "grid_thw": inputs["image_grid_thw"][0].tolist(),
                     "pixels": {"shape": list(pv.shape), "sum": float(pv.sum()), "sum_sq": float((pv * pv).sum()),
                                "head": pv.flatten()[:64].tolist(), "tail": pv.flatten()[-64:].tolist()}}
        else:
            chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok(chat).input_ids
        with torch.no_grad():
            probs = [p.probability for p in model.forward(options, app=t.app, task_family=t.family, ax_tree=t.ax_tree,
                                                          screenshot=shot, modality=a.modality, goal=goal)]
        gold = [l for l, o in zip(assignment.letters, options) if t.expected.get(o.element_id) == o.action]
        fixtures.append({"id": t.id, "family": t.family, "app": t.app, "goal": goal, "ax_tree": t.ax_tree,
                         "options": [o.__dict__ for o in options], "gold": gold,
                         "chat": chat, "input_ids": ids, "probs": probs, **extra})
        top = max(range(len(probs)), key=probs.__getitem__)
        print(f"{t.id}: {len(ids)} tokens, {len(options)} options, top {assignment.letters[top]} "
              f"p={probs[top]:.3f}, gold {''.join(gold)}, task {'ok' if task_correct(fixtures[-1]) else 'wrong'}", flush=True)
    json.dump({"run": run, "base": base, "modality": a.modality, "source": source,
               "letter_ids": [tok.encode(l, add_special_tokens=False)[0] for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
               "fixtures": fixtures}, open(a.out, "w"))
    if mm:   # the GUI-360 converter copies every converted step's screenshot, kept or not
        kept = {Path(f["screenshot"]).name for f in fixtures}
        for p in shots.iterdir():
            if p.name not in kept: p.unlink()
    print(f"{len(fixtures)} tasks, {summary(fixtures)} -> {a.out}")


def task_correct(f: dict, probs: list[float] | None = None) -> bool:
    """cua-bench-s1's task accuracy (eval.scoring.score_task): every element's likeliest action is its gold one."""
    probs = probs or f["probs"]
    best: dict[str, tuple[float, str]] = {}
    for p, o in zip(probs, f["options"]):
        if o["element_id"] not in best or p > best[o["element_id"]][0]: best[o["element_id"]] = (p, o["action"])
    gold = {f["options"]["ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(l)]["element_id"]: f["options"]["ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(l)]["action"]
            for l in f["gold"]}
    return all(best[e][1] == gold.get(e, "skip") for e in best)


def action_chosen(f: dict, probs: list[float] | None = None) -> bool:
    """The recorded action wins on its own element: the model would take it (whatever it decides elsewhere)."""
    probs = probs or f["probs"]
    for l in f["gold"]:
        k = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(l)
        if f["options"][k]["action"] != "skip":
            rivals = [p for p, o in zip(probs, f["options"]) if o["element_id"] == f["options"][k]["element_id"]]
            if probs[k] < max(rivals): return False
    return True


def summary(fixtures: list[dict], probs: list[list[float]] | None = None) -> str:
    probs = probs or [f["probs"] for f in fixtures]
    n = len(fixtures)
    return (f"task accuracy {sum(map(task_correct, fixtures, probs))}/{n}, "
            f"recorded action chosen {sum(map(action_chosen, fixtures, probs))}/{n}")


if __name__ == "__main__":
    main()
