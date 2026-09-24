# @ai-ecoverse/cua-s1.js

[Cua's cua-s1-forms](https://huggingface.co/cua-ai/cua-s1-forms) form-filling decision model in the browser, via
onnxruntime-web. For every field of a form it reads the field and a document's `Label: value` pairs and decides:
fill the field with one of those values, tick it, click it, or leave it alone. It has 706k parameters, the ONNX
graph is 3.1 MB, and one forward pass scores the whole form on the CPU.

The model, its training data and the planning rules are Cua's
([trycua/cua `libs/cua-s1`](https://github.com/trycua/cua/tree/main/libs/cua-s1), MIT). This package runs their
unmodified checkpoint and ports the code around it.

**[Live demo](https://ai-ecoverse.github.io/cua-s1.js/)** · **[cua-s1-4b demo](https://ai-ecoverse.github.io/cua-s1.js/4b.html)** · **[Weights](https://huggingface.co/ai-ecoverse/cua-s1.js)** · `npm install @ai-ecoverse/cua-s1.js onnxruntime-web`

![cua-s1.js demo](docs/demo.png)

```ts
import * as ort from "onnxruntime-web/wasm";
import { loadCuaS1, extractEntities } from "@ai-ecoverse/cua-s1.js";

const model = await loadCuaS1("https://huggingface.co/ai-ecoverse/cua-s1.js/resolve/main/cua-s1-forms", { ort });
const plan = await model.plan(
  "Northwind Clinic - New Patient Registration",
  [
    { role: "Edit", label: "Phone number", value: "", token: "phone" },
    { role: "CheckBox", label: "I consent to treatment", checked: false, token: "consent" },
    { role: "Button", label: "Submit", token: "submit" },
  ],
  extractEntities("Tel: (503) 555-0142\nWork phone: (503) 555-0110"),
  { minConfidence: 0.5, allowSubmit: false },
);
plan.decisions;   // one per element: action, entityIndex, probability, distribution
plan.actions;     // what to execute, in order: fills, checkboxes, then at most one submit click
```

Nothing is executed by the library: `plan()` returns decisions, and the caller applies them. The loader checks the
graph's SHA-256 against the manifest.

## How It Works

- **Export** (`export/export.py`): pins the Hugging Face repo to a commit, downloads the safetensors and JSON sidecar
  at that commit, and loads them with `cua_s1.model.load_checkpoint`, which validates the SHA-256 tensor signature.
  It then exports the model with the `torch.export`-based ONNX exporter, with dynamic batch, context, option and
  option-token axes. The legacy tracer bakes the sample's sequence length into `nn.MultiheadAttention`'s reshapes,
  so it can't be used here. The graph takes the tensors `ByteCollator` produces and returns logits and
  probabilities. A second graph runs the same modules for a batch whose rows share one option list, which every
  form plan is: it encodes the options once and broadcasts them to every element, where upstream's `forward`
  encodes them again for each row. `loadCuaS1` uses it when the manifest names one (`sharedOptions: false` opts
  out).
- **Verification:** the export scores 1,048 decisions from 40 synthetic episodes (`cua_s1.synth.episode_rows`, the
  training generator) in mixed batches with both PyTorch and onnxruntime. The largest probability difference is
  3.2e-6, with no argmax flips. The shared-options graph, one episode per batch, is held to upstream's unmodified
  PyTorch `forward` with the same result, and in onnxruntime it matches the per-row graph exactly. The PyTorch
  outputs are the fixtures for the JS tests, which run both graphs.
- **Runtime** (`src/`): a port of `render_context`, `render_options`, `ByteCollator`, the `Label: value` line parser
  from `pdf.py`, and the model-independent planner rules (actionable roles, title normalization, ordering, and the
  one-submit-click rule). The model only ever sees strings, so these are reproduced exactly: Python's code-point
  slicing, UTF-8 truncation by bytes, `errors="replace"` for lone surrogates, and `splitlines()`'s separators.
  `test/schema.test.ts` checks all of it against Python's output (`export/schema_fixtures.py`).
- **Demo** (`demo/`): reads a real HTML form from the DOM and maps text inputs and textareas to `Edit`, checkboxes
  to `CheckBox` and buttons to `Button`, the three roles cua-s1 was trained on. Selects and radio groups are left
  alone. It shows the plan, and fills the DOM only when you press Apply. PDF text comes from pdf.js.

## Results

On the demo's clinic form (15 elements, a document with 15 `Label: value` pairs including look-alikes such as
`Tel` / `Work phone` / `Emergency contact phone`), all 11 fields were filled with the right values. Consent was
ticked and the newsletter box left alone. The submit click only runs with `allowSubmit`.

- **Latency:** about 60 ms per plan on one WASM thread in a visible Chrome tab on an M4 Max, with the options
  encoded once per plan. The per-row graph, which re-encodes the 18 options for all 15 elements (270 option passes
  instead of 18), takes 160 ms. That is a 2.7× speedup, and the decisions are identical. It grows with the number of
  fields. The model loads in about 200 ms.

On the synthetic episodes, the PyTorch model itself picks the labelled option 98.7% of the time (1,034 of 1,048).
The model card reports 99.95% on its own form-disjoint test split. The misses are mostly conservative:
13 of 440 fills came out as `skip`, and one filled the wrong entity (a portfolio URL field took the LinkedIn URL,
p = 0.78). Keep a confidence threshold and look at the plan before applying it.

A threshold does not catch forms outside the training vocabulary. The synthetic forms draw every field label from
the 55 concepts in upstream's `cua_s1/concepts.py`, and on labels outside that catalogue (logistics, DevOps and
veterinary forms, and one Chinese form) the checkpoint scored 12 of 41 and answered `skip` for 36 of the 41 elements
at a mean confidence of 0.974 ([trycua/cua#3978](https://github.com/trycua/cua/issues/3978)). The form's own field
labels matter most: with the same structure and values but catalogue phrases as labels, 11 of 11 were right. A plan that
skips nearly every field on an unfamiliar form is the model being out of scope, not a form with nothing to fill.

## cua-s1-4b

[`cua-ai/cua-s1-4b-0.2`](https://huggingface.co/cua-ai/cua-s1-4b-0.2) is a different kind of model: a LoRA on
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), trained by Cua with supervised fine-tuning and then RLOO on live
computer-use tasks. It is given a goal, one screen and a closed list of (element, action) options, one letter each,
and scores the options in one forward pass: a softmax over the option letters' logits at the final position. This is
a port of Cua's `cua_s1/four_b.py`. Cua trained two independent adapters, and each is its own bundle:
`cua-s1-4b-0.2` reads the accessibility tree, `cua-s1-4b-0.2-multimodal` a screenshot (`{ screenshot: ImageData }`
instead of `axTree`) through Qwen3.5's vision tower.

```ts
import * as ort from "onnxruntime-web/webgpu";
import { elementDecisions, loadCuaS1FourB } from "@ai-ecoverse/cua-s1.js/4b";

const model = await loadCuaS1FourB("https://huggingface.co/ai-ecoverse/cua-s1.js/resolve/main/cua-s1-4b-0.2", { ort });
const r = await model.score(
  [
    { elementId: "el_0", role: "Edit", label: "Search", action: "skip" },
    { elementId: "el_1", role: "Button", label: "Bold", action: "skip" },
    { elementId: "el_1", role: "Button", label: "Bold", action: "click" },
    { elementId: "el_2", role: "Button", label: "Italic", action: "skip" },
    { elementId: "el_3", role: "Button", label: "Save", action: "skip" },
  ],
  {
    app: "word", taskFamily: "multi_step_submit", goal: "Make the selected heading bold.",
    axTree: '- Edit "Search" @ [300, 8, 620, 32]\n- Button "Bold" @ [104, 108, 128, 132]\n'
      + '- Button "Italic" @ [128, 108, 152, 132]\n- Button "Save" @ [8, 8, 32, 32]',
  },
);
r.options;                  // A 0.052, B 0.000, C 0.736, D 0.124, E 0.088
elementDecisions(r.options); // per element, its likeliest action: click "Bold", skip the rest
```

Read the result per element. Cua's training target spreads the probability over every acceptable option, and that
includes every correct `skip`. So the single top option says less than the comparison on each element between its
action and its `skip`, which is how cua-bench-s1 scores the model and what `elementDecisions` returns.

- **Export** (`export/build_4b_model.sh`): folds the adapter into the base in fp32, builds the graph with the
  onnxruntime-genai model builder (int8 weights, fp32 activations, WebGPU), shards it into files of at most 32 MB,
  and scores the fixtures. This is kev.js's pipeline: Kev is also a LoRA on Qwen3.5. The graph returns hidden
  states. Qwen3.5-4B ties its output layer to the embeddings, so the head is just the 26 letters' embedding rows in
  fp32 (266 kB). The bundle is 4.7 GB.
- **Screenshots** (`export/four_b/vision.py`, `src/four-b-vision.ts`): the multimodal adapter also adapts the vision
  tower, so it is exported from the same merge: Qwen3.5's 24-block ViT and patch merger. Everything that depends on
  the image's size is computed in JS and passed in: the bilinear taps into the 48×48 position table, the 2D rotary
  angles, and the 3D (mRoPE) positions of the image tokens. That leaves plain tensor math in the graph, which is
  checked against transformers' own vision module. The decoder takes the result as `image_embeds` at the
  `<|image_pad|>` tokens. `preprocess` ports Qwen's image processor, including torch's antialiased bicubic resize
  (Pillow's a = −0.5 kernel). The vision graph keeps its weights in fp16 (640 MB): int8 weights put its output 4.6%
  off in RMS and doubled the decoder's error. The bundle is 5.0 GB.
- **Test screens** (`export/four_b/fixtures.py`): real Word, Excel and PowerPoint steps from
  [GUI-360](https://huggingface.co/datasets/vyokky/GUI-360)'s test split (MIT), converted with cua-bench-s1's own
  `datagen.gui360`: the live UI Automation tree (or the screenshot), up to 16 elements, `skip` for each and the
  recorded action for the one that was acted on, and the goal (the episode's request and the step's subtask) that
  upstream's evaluation states in the prompt. Every task outside `multi_step_submit` is kept and every tenth
  `multi_step_submit` task, 60 in all. 0.2's training mix includes GUI-360 screens, and Cua does not publish which
  episodes it held out, so these tasks measure the port rather than 0.2's generalization. They replace cua-bench-s1's
  synthetic screens, which 0.1 was trained on and 0.2 never saw: Cua's benchmark has 0.2's screenshot adapter at or
  below chance on most of those families.
- **Prompt:** `renderChat` reproduces `build_prompt` and Qwen's chat template, which ends in an open `<think>`
  block because upstream leaves thinking on. On every fixture task the text matches byte for byte, goal included,
  and the tokenizer (`@huggingface/tokenizers`) reproduces upstream's token ids. One deliberate difference: `<|...|>`
  in caller text is made inert, so a page's labels cannot inject chat tokens. `renderAxTree` writes cua-bench-s1's
  synthetic accessibility tree, for screens that don't come with one.

| | tree (60 GUI-360 steps) | screenshot (the same 60) |
|---|---|---|
| Cua's `FourBModel`, fp32 PyTorch: task accuracy | 51/60 | 31/60 |
| fp32 graphs vs `FourBModel`: max \|Δp\| | 1.2e-4, no argmax flips | 9.3e-4, no argmax flips |
| int8 bundle on onnxruntime CPU | 0.062, 3 flips, 51/60 | 0.10, no flips, 31/60 |
| int8 bundle in Chrome on WebGPU | 0.031, 1 flip, 51/60 | 0.038, no flips, 31/60 |
| latency in Chrome on WebGPU, M4 Max | 860–1,360 ms (median 1,020) for 700–1,090 tokens | 2,160–2,440 ms (median 2,250) for 1,030–1,240 tokens, vision tower included |

Task accuracy is cua-bench-s1's: every element's likeliest action is the expected one. Quantization moves 0.2's
probabilities more than 0.1's (0.1's text bundle stayed within 0.005) because 0.2's are far less flat, but on these tasks it
changes no decision the metric counts. A bundle loads in 7–9 s once it is local; the first visit downloads ~5 GB
into Cache Storage. With more than 26 options, split the screen.

The screenshot adapter is the weaker one on these screens, and that is the adapter, not the port: Cua's own
`FourBModel` gets the same 31 of 60. On 28 of the 29 it misses it acts on nothing, preferring `skip` on the element
that was acted on; on the other it ticks one checkbox too many. Cua reports 0.93 for it on its own screenshot split,
which is not published; its evaluation harness is not in the repository either. The text adapter is the one to use
when the page gives you an accessibility tree.

Cua's [benchmark](https://github.com/trycua/cua/tree/main/libs/cua-bench-s1) reports 0.2 at 0.83–1.00 per family on
its held-out GUI-360 split (text; pagination, one mislabelled family, aside) and 0.93 overall with screenshots, against 0.17–0.57 for 0.1, and 0.94 of live
multi-step tasks completed from the accessibility tree. The adapters are Apache-2.0, like Qwen3.5-4B.

## Development

```bash
git clone --recursive https://github.com/ai-ecoverse/cua-s1.js && cd cua-s1.js
cd export && uv sync && uv run python export.py --out ../public/models/cua-s1-forms
uv run python schema_fixtures.py > ../fixtures/schema.json
cd .. && npm install && npm test && npm run dev     # http://127.0.0.1:5174
```

`vendor/cua` is a sparse submodule of trycua/cua (`libs/cua-s1` and `libs/cua-bench-s1`:
`git -C vendor/cua sparse-checkout set libs/cua-s1 libs/cua-bench-s1`), so the export and the fixtures always run
Cua's own code. `export.py` pins the Hugging Face checkpoint to a commit and writes the bundle under `r-<commit>/`
with a `manifest.json`. `HF_TOKEN=... uv run python upload_hf.py` publishes it to `ai-ecoverse/cua-s1.js`, the
revision's files and the manifest in one commit per model (`--only <folder>` publishes one).

cua-s1-4b needs the `four-b` dependency group and about 45 GB of disk at peak (60 GB for screenshots):
`uv sync --group four-b && ./build_4b_model.sh [--multimodal] cua-ai/cua-s1-4b-0.2`. It writes
`public/models/cua-s1-4b-0.2[-multimodal]` (not in git) and the fixtures (`fixtures/cua-s1-4b-0.2*.json`, plus the
GUI-360 screenshots; the first multimodal build downloads GUI-360's 6.3 GB of test images). `npm test` checks the prompt rendering and the image preprocessing against the fixtures, and
checks the tokenizer and the graphs as well when a bundle is there. `npm run dev` serves both demos: `/` and
`/4b.html` (`?modality=multimodal` for screenshots).

Releases publish to npm from CI (`.github/workflows/release.yaml`, trusted publishing, OIDC) via
[semantic-release](https://semantic-release.org/) on every push to `main`. Commits that follow
[Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `BREAKING CHANGE:`) cut the next
patch / minor / major; `v0.1.1` is the last hand-tagged release. The demo deploys to GitHub Pages on every push to
`main` and loads the weights from Hugging Face.

## Related

- [kev.js](https://github.com/ai-ecoverse/kev.js): Kev decision models in the browser, whose export pipeline the
  cua-s1-4b bundles are built with. Its `-vision` bundles answer questions about images through the same Qwen3.5
  vision tower.
- [jev-omni.js](https://github.com/ai-ecoverse/jev-omni.js): the Gemma 4 12B Jev-Omni decision classifier on WebGPU.
- [decision-vision-bench](https://github.com/ai-ecoverse/decision-vision-bench): cua-s1-4b-0.2 multimodal, Kev vision
  and Jev-Omni on one mixed image decision set, including this repo's GUI-360 screens.

## License

MIT, like cua-s1. The ported schema and planner logic keep Cua AI, Inc.'s copyright notice (`LICENSE`), and the
checkpoint is MIT per its [model card](https://huggingface.co/cua-ai/cua-s1-forms). cua-s1 is a research
checkpoint trained on synthetic forms: read its model card and
[SECURITY.md](https://github.com/trycua/cua/blob/main/libs/cua-s1/SECURITY.md) before using it on anything that
matters.
