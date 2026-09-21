# @ai-ecoverse/cua-s1.js

[Cua's cua-s1-forms](https://huggingface.co/cua-ai/cua-s1-forms) form-filling decision model in the browser, via
onnxruntime-web. For every field of a form it reads the field and a document's `Label: value` pairs and decides:
fill the field with one of those values, tick it, click it, or leave it alone. It has 706k parameters, the ONNX
graph is 3.3 MB, and one forward pass scores the whole form on the CPU.

The model, its training data and the planning rules are Cua's
([trycua/cua `libs/cua-s1`](https://github.com/trycua/cua/tree/main/libs/cua-s1), MIT). This package runs their
unmodified checkpoint and ports the code around it.

```ts
import * as ort from "onnxruntime-web/wasm";
import { loadCuaS1, extractEntities } from "@ai-ecoverse/cua-s1.js";

const model = await loadCuaS1("/models/cua-s1-forms", { ort });
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

Nothing is executed by the library: `plan()` returns decisions, and the caller applies them.

## How It Works

- **Export** (`export/export.py`): pins the Hugging Face repo to a commit, downloads the safetensors and JSON sidecar
  at that commit, and loads them with `cua_s1.model.load_checkpoint`, which validates the SHA-256 tensor signature.
  It then exports the model with the `torch.export`-based ONNX exporter, with dynamic batch, context, option and
  option-token axes. The legacy tracer bakes the sample's sequence length into `nn.MultiheadAttention`'s reshapes,
  so it can't be used here. The graph takes the tensors `ByteCollator` produces and returns logits and
  probabilities.
- **Verification:** the export scores 1,048 decisions from 40 synthetic episodes (`cua_s1.synth.episode_rows`, the
  training generator) in mixed batches with both PyTorch and onnxruntime. The largest probability difference is
  3.2e-6, with no argmax flips. The PyTorch outputs are the fixtures for the JS tests.
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

- **Latency:** 168 ms per plan on one WASM thread in Chrome on an M4 Max. The model loads in about 240 ms.
- **Where the time goes:** every element carries the same options, and the unmodified graph re-encodes them for
  every element (285 option passes here). Encoding the options once per plan would be a 10×+ speedup, but needs a
  graph that differs from upstream's.

On the synthetic episodes, the PyTorch model itself picks the labelled option 98.7% of the time (1,034 of 1,048).
The model card reports 99.95% on its own form-disjoint test split. The misses are mostly conservative:
13 of 440 fills came out as `skip`, and one filled the wrong entity (a portfolio URL field took the LinkedIn URL,
p = 0.78). Keep a confidence threshold and look at the plan before applying it.

## Development

```bash
git clone --recursive https://github.com/ai-ecoverse/cua-s1.js && cd cua-s1.js
cd export && uv sync && uv run python export.py --out ../public/models/cua-s1-forms
uv run python schema_fixtures.py > ../fixtures/schema.json
cd .. && npm install && npm test && npm run dev     # http://127.0.0.1:5174
```

`vendor/cua` is a sparse submodule of trycua/cua (`libs/cua-s1` only), so the export and the fixtures always run
Cua's own code.

## License

MIT, like cua-s1. The ported schema and planner logic keep Cua AI, Inc.'s copyright notice (`LICENSE`), and the
checkpoint is MIT per its [model card](https://huggingface.co/cua-ai/cua-s1-forms). cua-s1 is a research
checkpoint trained on synthetic forms: read its model card and
[SECURITY.md](https://github.com/trycua/cua/blob/main/libs/cua-s1/SECURITY.md) before using it on anything that
matters.
