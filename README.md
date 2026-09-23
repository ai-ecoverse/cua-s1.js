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

[`cua-ai/cua-s1-4b-0.1`](https://huggingface.co/cua-ai/cua-s1-4b-0.1) is a different kind of model: a LoRA on
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). It is shown one screen as an accessibility tree and a closed
list of (element, action) options, one letter each, and picks the single best next action. It is a port of Cua's
`cua_s1/four_b.py`: one forward pass, then a softmax over the option letters' logits at the final position. Text
modality only; the multimodal adapter would also need the vision tower.

```ts
import * as ort from "onnxruntime-web/webgpu";
import { loadCuaS1FourB, renderAxTree } from "@ai-ecoverse/cua-s1.js/4b";

const model = await loadCuaS1FourB("https://huggingface.co/ai-ecoverse/cua-s1.js/resolve/main/cua-s1-4b-0.1", { ort });
const r = await model.score(
  [
    { elementId: "el_0", role: "Edit", label: "Email", action: "fill", entityId: "ent_0" },
    { elementId: "el_0", role: "Edit", label: "Email", action: "skip" },
    { elementId: "el_1", role: "Button", label: "Submit", action: "click" },
  ],
  {
    app: "clinic_intake", taskFamily: "form_filling",
    axTree: renderAxTree("Clinic intake", [
      { id: "el_0", role: "Edit", label: "Email", value: "", required: true },
      { id: "el_1", role: "Button", label: "Submit" },
    ], [{ id: "ent_0", label: "E-mail", value: "amara@example.invalid" }]),
  },
);
r.options; // A fill 0.294, B skip 0.374, C click 0.332: r.best is B. 0.1 is this flat, and here it is wrong
```

- **Export** (`export/build_4b_model.sh`): folds the text adapter into the base in fp32, builds the graph with the
  onnxruntime-genai model builder (int8 weights, fp32 activations, WebGPU), shards it into files of at most 32 MB,
  and scores the fixtures. This is kev.js's pipeline: Kev is also a LoRA on Qwen3.5. The graph returns hidden
  states. Qwen3.5-4B ties its output layer to the embeddings, so the head is just the 26 letters' embedding rows in
  fp32 (266 kB). The bundle is 4.7 GB.
- **Prompt:** `renderChat` reproduces `build_prompt` and Qwen's chat template, which ends in an open `<think>`
  block because upstream leaves thinking on. On the 57 fixture tasks the text matches byte for byte, and the
  tokenizer (`@huggingface/tokenizers`) reproduces upstream's token ids. One deliberate difference: `<|...|>` in
  caller text is made inert, so a page's labels cannot inject chat tokens. `renderAxTree` writes cua-bench-s1's
  accessibility tree, including the goal and source record that fill options point into.
- **Parity** against Cua's `FourBModel` in fp32 PyTorch, on 57 tasks from cua-bench-s1's own generator (19 apps,
  7 families): the fp32 graph is within 7.5e-5. The int8 graph is within 0.0048 on onnxruntime, with the argmax
  changed on 4 near-ties. In Chrome on WebGPU it is within 0.0042, with 2 flips, both where upstream's top two
  options are within 0.0002 of each other.
- **Latency:** 480–1,270 ms per decision (median 680 ms) for 358–979-token prompts on WebGPU in Chrome on an M4 Max.
  The bundle loads in about 6 s once it is local; the first visit downloads 4.7 GB into Cache Storage.

What the checkpoint itself does: its choices are nearly flat (mean top probability 0.21 over 3–24 options). Its
top pick is an option the answer key accepts on 46 of the 57 tasks, where a uniform pick would manage 36. Cua's
training target spreads over every acceptable option, including every correct `skip`. Cua's own
[benchmark](https://github.com/trycua/cua/tree/main/libs/cua-bench-s1) has 0.1 at 0.167–0.571 per family on
held-out real screens.

It was trained on option lists in which every empty field has one `fill` option that already names the right value
from the source record. So it decides what to do next, not which value goes where. With more than 26 options, split
the screen. Cua publishes the adapter without a license file; the weights here are for research and evaluation.

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

cua-s1-4b needs the `four-b` dependency group and about 45 GB of disk at peak: `uv sync --group four-b &&
./build_4b_model.sh cua-ai/cua-s1-4b-0.1`. It writes `public/models/cua-s1-4b-0.1` (not in git) and
`fixtures/cua-s1-4b-0.1.json`. `npm test` checks the prompt rendering against the fixtures, and checks the tokenizer
and the graph as well when the bundle is there. `npm run dev` serves both demos: `/` and `/4b.html`.

Releases publish to npm from CI (`.github/workflows/release.yaml`, trusted publishing, OIDC) via
[semantic-release](https://semantic-release.org/) on every push to `main`. Commits that follow
[Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `BREAKING CHANGE:`) cut the next
patch / minor / major; `v0.1.1` is the last hand-tagged release. The demo deploys to GitHub Pages on every push to
`main` and loads the weights from Hugging Face.

## License

MIT, like cua-s1. The ported schema and planner logic keep Cua AI, Inc.'s copyright notice (`LICENSE`), and the
checkpoint is MIT per its [model card](https://huggingface.co/cua-ai/cua-s1-forms). cua-s1 is a research
checkpoint trained on synthetic forms: read its model card and
[SECURITY.md](https://github.com/trycua/cua/blob/main/libs/cua-s1/SECURITY.md) before using it on anything that
matters.
