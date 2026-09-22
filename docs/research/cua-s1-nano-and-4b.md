# Survey: `cua-s1-nano-0.1` and `cua-s1-4b` (0.1, 0.2)

Research notes, 2026-09-22. No code changes: this documents what three undocumented Cua checkpoints
are, what was measured, and what is still unknown. Nothing here is wired into the package yet.

Cua published the three new checkpoints with no model card, no tokenizer, no prompt format, and no
supporting code (`cua-s1-forms`, listed below for comparison, does have a card).
`trycua/cua@origin/main` has no commits touching `libs/cua-s1` since
[`9bbfa7dd3`](https://github.com/trycua/cua/commit/9bbfa7dd3) (the commit `vendor/cua` is pinned to),
and the whole `libs/cua-s1` tree contains no mention of `cua-s1-nano`, `cua-s1-4b`, `siglip`,
`vision_backbone` or `visual_proj`. There are no Hugging Face discussions on any `cua-ai` repo, so
none of this has been asked publicly.

| repo | revision | published | what it is |
| --- | --- | --- | --- |
| [`cua-ai/cua-s1-forms`](https://huggingface.co/cua-ai/cua-s1-forms) | `f54adbf` | 2026-09-19 | the 706k byte scorer this package already ships |
| [`cua-ai/cua-s1-nano-0.1`](https://huggingface.co/cua-ai/cua-s1-nano-0.1) | `1f93fd0` | 2026-09-21 19:36Z | 855k scorer, `text/` and `multimodal/`, SigLIP projection |
| [`cua-ai/cua-s1-4b-0.1`](https://huggingface.co/cua-ai/cua-s1-4b-0.1) | `88d8b8a` | 2026-09-21 22:36Z | LoRA on Qwen3.5-4B; measurably a bad run |
| [`cua-ai/cua-s1-4b-0.2`](https://huggingface.co/cua-ai/cua-s1-4b-0.2) | `4b1cb70` | 2026-09-22 03:46Z, updated 07:09Z | same architecture retrained; works |

## `cua-s1-nano-0.1`

Two variants, `text/` and `multimodal/`, each 3.43 MB and **855,296 parameters** (forms is 706k /
2.83 MB). Both declare `format: cua-s1`, `format_version: 1`.

```
text/       {"context_tokens": 256, "option_tokens": 96, "rank": 128, "width": 128}
multimodal/ {"context_tokens": 256, "option_tokens": 96, "rank": 128, "width": 128,
             "vision_backbone": "siglip"}
```

It is `tinyx` rearranged, not a new family:

| | `cua-s1-forms` (`tinyx`) | `cua-s1-nano-0.1` |
| --- | --- | --- |
| config `encoder` | `"tinyx"` | **absent** |
| context / option tokens | 224 / 96 | 256 / 96 |
| width, rank, heads, FFN | 128, 128, 4, 512 | identical |
| byte embedding | one shared `embedding` + `position` | **split** per stream: `text_context_encoder.{embedding,position}` (256 positions) and `option_encoder.{embedding,position}` (96) |
| encoder depth | 2 context layers, 1 option layer | identical |
| head | `AttentionHead` (`query`/`key`/`value` + `context_norm`/`option_norm`) | identical |
| vision | none | `visual_proj` = `LayerNorm(768)` then `Linear(768 -> 128)` |

### It contains no vision tower

The entire visual path is four tensors:

```
visual_proj.0.{weight,bias}  F32 [768]        LayerNorm(768)
visual_proj.1.{weight,bias}  F32 [128, 768]   Linear(768 -> 128)
```

768 is SigLIP's *base*-class vision width, so nano consumes **externally computed SigLIP embeddings**
and projects them into its 128-wide space. The image encoder is the caller's problem, which is good
news for a browser port: the vision half already exists as ONNX.

### Measured facts

Recomputing `cua_s1.checkpoint._state_signature` (sha256 over canonical-JSON config, then per tensor
sorted by name its `[name, dtype, shape]` followed by raw bytes) reproduces both published digests
exactly, so the tensor inventory below is provably complete:

```
text/        7fc7c402e7a1b07bf86f0cfccb2aa952b00935be1c2b455016c4bf0620eb8261  MATCH
multimodal/  2f014b1c4279ad54d5989c2abf5260584c2116cb9212b9e5b775e680e1a5cb6a  MATCH
```

**`text/`'s `visual_proj` is at exact PyTorch initialization and was never trained.** LayerNorm
weight is uniformly 1.0 (std 0.0) and bias uniformly 0.0; the Linear is uniform within
±0.0361, which is the default bound 1/sqrt(768) = 0.03608. So the text variant provably ignores
images and carries roughly 99k dead parameters (about 0.4 MB of its 3.43 MB).

**`multimodal/`'s `visual_proj` is trained.** LayerNorm weight 0.9910 ± 0.0177, deviating from init
by up to 0.0324, with a nonzero bias (max |b| 0.0286); the Linear has std 0.0264 and reaches 0.0665,
past the init bound. The vision path is real.

**The two variants are independent training runs.** All 51 shared tensor names differ; none are
bit-identical. `multimodal/` is not `text/` plus a trained projection, and no trunk is shared, so
one cannot be derived from the other.

### What is still unknown

1. **The fusion rule.** Renaming `encoder` to `text_context_encoder` implies the `AttentionHead`
   attends over text context tokens plus projected visual tokens, but nothing in the checkpoint says
   whether they are concatenated, prepended or summed. There is no vision position table, and
   `text_context_encoder.position` is exactly `context_tokens` (256), which hints the visual tokens
   sit outside that budget and carry no positional embedding. That is an inference, not a fact.
2. **Pooled or per-patch.** `LayerNorm` then `Linear` works identically for one pooled embedding or
   for all patch tokens. 196 patches (224px) would crowd a 256-token context, so pooled-per-image is
   more plausible for an 855k-parameter model, but this is a guess.
3. **Which SigLIP.** See below. Unanswerable from the weights.
4. **Upstream cannot load it.** `cua_s1.model.make_system` dispatches on `config["encoder"]` and
   raises `ValueError("model config field 'encoder' must be 'tiny' or 'tinyx'")`. With no `encoder`
   key, this checkpoint is rejected, so `export/export.py` cannot export it: pointing `--repo` at
   nano is not sufficient, a new model class is required first.

Guessing 1 through 3 would produce a demo that looks right and is silently miscalibrated, which is
the failure mode this package's SHA-pinned revisions and PyTorch parity fixtures exist to prevent.

### SigLIP candidate set

Nano's 768-wide projection input pins the encoder to SigLIP's base class. Verified vision widths:

| family | vision `hidden_size` | verdict |
| --- | --- | --- |
| `siglip-base-patch16-*`, `siglip2-base-patch16-*` | **768** (12 layers, FFN 3072, 12 heads) | candidate |
| `siglip-large-patch16-*` | 1024 | excluded |
| `siglip-so400m-patch14-*`, `siglip2-so400m-*` | 1152 | excluded |

That leaves nine candidates: SigLIP v1 and v2 at 224 / 256 / 384 / 512, plus `siglip2-base-patch16-naflex`.
Resolution and v1-versus-v2 leave no trace in a 768 -> 128 projection, so the weights cannot narrow
this further. Picking wrong yields plausible but miscalibrated probabilities.

### Browser budget, if the contract were known

| piece | source | size |
| --- | --- | --- |
| SigLIP-base vision tower | `Xenova/siglip-base-patch16-224`, `onnx/vision_model_int8.onnx` | 94.1 MB (fp32 371.8, fp16 186.1, q4 63.3) |
| nano multimodal scorer | `cua-ai/cua-s1-nano-0.1` | 3.43 MB |

About 97 MB for a screenshot-conditioned decision model, against 822 MB for Kev-0.8B in
[`kev.js`](https://github.com/ai-ecoverse/kev.js). The vision half is already ONNX and already
packaged for `transformers.js`, so the only missing piece once Cua answers is a concatenation and one
`Linear`.

## `cua-s1-4b` (0.1 and 0.2)

PEFT LoRA adapters, `peft_version` 0.18.1, r=16, alpha=32, dropout 0.05, `task_type: CAUSAL_LM`,
base `Qwen/Qwen3.5-4B`. Both releases ship `text/` (85.0 MB, 21.2M params) and `multimodal/`
(101.7 MB, 25.4M params). 0.1 additionally duplicates `text/` at the repo root; 0.2 drops that.

### The base is confirmed, by construction

`Qwen/Qwen3.5-4B` is itself multimodal (`Qwen3_5ForConditionalGeneration`, vision tower depth 24,
hidden 1024, FFN 4096, `out_hidden_size` 2560). Loading it with `AutoModelForImageTextToText` and
mapping every adapter tensor to a module path resolves **178/178** targets with no missing or extra
entries, and the injected LoRA layer count matches. So the declared base is correct, and the
`text/` versus `multimodal/` split is exactly whether the vision tower is adapted.

LoRA placement, from the tensor names:

- MLP (`gate_proj`, `up_proj`, `down_proj`) on **all 32** language layers.
- Attention (`q`, `k`, `v`, `o_proj`) on only **8** layers, indices 3 to 31, i.e. every fourth.
- `multimodal/` adds `visual.blocks.{0..23}.mlp.linear_fc{1,2}` and `visual.merger.linear_fc{1,2}`,
  and prefixes the language stack with `language_model.`.

0.2 is a genuine retrain: semantically identical configs (the only diff is `target_modules`
serialization order), identical file sizes, different content hashes
(`multimodal/adapter_model.safetensors` sha256 `c3973b09dd…` in 0.1 versus `38ecd5a919…` in 0.2).

### The contract: a whole-form plan

The model takes the form's elements plus the document's `Label: value` pairs in one chat turn and
emits one line per element. A system message is required; without it both releases fall back to
prose and markdown tables. Measured against `cua_s1.synth.episode_rows` labels, 94 decisions over
4 episodes (seeds 10000-10003), greedy decoding, `multimodal/` adapter:

| adapter | lines parsed | action accuracy | exact (action + entity) |
| --- | --- | --- | --- |
| **0.2** | **94/94** | **0.840** | **0.809** |
| none (base Qwen3.5-4B) | 87/94 | 0.713 | 0.681 |
| 0.1 | 69/94 | 0.532 | 0.500 |

0.1 is a dud: it shifts the answer distribution hard onto short label tokens (KL(adapter‖base) up to
3.86 on an enumerated-options prompt, ~0.65 of the mass on digits) while scoring **below** the
unmodified base, and under every per-element pointer scaffold tried it sat at chance (best 2/17
versus 1/17 for the base, mean p(label) ~0.07 against 0.056 chance). 0.2 barely perturbs the base
(KL 0.012 to 0.568) yet adds 12.8 points and parses every line.

It is **not** a fixed-protocol decision model. The output shape follows the instruction line:

```
"Reply with one line per element, \"<index>: fill <Label>: <value>\" or \"<index>: <check|click|skip>\"."
  -> 0: fill Company: Example Industries
     9: check Consent to electronic communication (required)
     12: click Complete registration
```

Echo the element syntax instead and it returns `0: ELEMENT Edit "Company" value="Example Industries"`.
The format comes from the base's instruction following; the LoRA contributes task skill (label to
entity matching), not a template. That is a different shape of thing from this package's closed-set
argmax over rendered options.

### Scaffolds ruled out

Raw (non-chat) completion, JSON in and JSON out, unenumerated chat prompts (which trigger safety
refusals: "appears to be a phishing attempt"), per-element single-token label pointers in both digit
and letter styles, and `element_token` addressing. The last is ruled out by reading upstream:
`element_token` is a snapshot-bound execution handle, and `schema.Element` documents `index` as
observation metadata that "must not be used to target an action", so neither enters the model's input.

### Images strengthen the 0.1 delta

With a rendered form screenshot attached, 0.1's KL(adapter‖base) rises from 2.007 to **4.515**, and
about 0.62 of the mass lands on option labels while the same prompt without the adapter is 84% prose.
Whatever 0.1 learned, the vision LoRA is load-bearing rather than decoration. This was not re-run for
0.2.

## Comparison, and what it implies for this package

On the same synthetic generator:

| model | parameters | exact accuracy |
| --- | --- | --- |
| `cua-s1-forms` byte scorer | **706k** | **98.7%** (1,034/1,048, per this repo's README) |
| `cua-s1-4b-0.2`, whole-form plan | 4B | 0.809 |
| `Qwen3.5-4B` zero-shot, same prompt | 4B | 0.681 |

Caveat: these are not measured the same way. The scorer picks from a closed option set, whereas the
4B must generate the right value string which is then matched back to an entity, so 0.809 is a
harsher metric than 98.7%. The 4B numbers also rest on 94 decisions, enough to separate 0.50 from
0.81 but not to split hairs. Even allowing for that, the gap is wide: the 706k graph this package
already ships outperforms a 4B model at roughly 1/5700th the parameters.

Consequences:

- **The 4B is not worth porting for form filling.** It would be ~4.7 GB at q8f32 (the size
  `kev.js` already handles for Kev-4B on WebGPU) to get a worse form filler than the 3.3 MB graph
  here. Its interest is as a general instruction-following UI agent, not as a `cua-s1-forms`
  replacement.
- **Nano multimodal is the route worth taking.** About 97 MB total, the vision half already ONNX,
  and the vision projection now proven to be trained. It is blocked only on the fusion rule and the
  SigLIP variant.
- **`export/export.py` cannot target nano** until upstream grows a model class for an `encoder`-less
  config, or this repo defines one and accepts that it has no reference implementation to verify
  against.

## Questions for Cua

1. `cua-s1-nano-0.1` ships no `encoder` key, so `cua_s1.model.make_system` rejects it. What class
   loads it, and will that land in `libs/cua-s1`?
2. How are `visual_proj` outputs fused into the context: concatenated to the encoded text context,
   prepended, or summed? Do they receive position embeddings?
3. Pooled image embedding or per-patch tokens, and how many?
4. Which SigLIP checkpoint produces the 768-d input (v1 or v2, and which resolution)?
5. Is `cua-s1-4b-0.1` deprecated in favour of 0.2? Its measured behaviour is worse than the
   unmodified base.
6. Is there a documented prompt format for `cua-s1-4b`, or is instruction following the contract?

## Reproducing

The probe harness is not committed with these notes. It loads the base with
`AutoModelForImageTextToText`, asserts every adapter tensor maps to a real module and that the
injected LoRA layer count matches (a silently unapplied adapter would otherwise produce
authoritative-looking nonsense), then compares adapter-enabled against `model.disable_adapter()` on
the same weights. Ground truth comes from `cua_s1.synth.episode_rows`, and `cua_s1.schema`'s
`render_context` / `render_options` supply the input strings. Environment: `transformers` 5.17,
`peft` 0.21, `torch` 2.8 on MPS, bf16, greedy decoding.

Two measurement notes for anyone redoing this:

- With 18 options the index `"11"` shares a first token with `"1"`, so first-token scoring cannot
  separate them; score whole continuations.
- Relative-difference comparisons against `text/`'s `visual_proj.0.bias` are meaningless because it
  is all zeros, so the denominator vanishes.
