// cua-s1-4b in the browser: a port of cua_s1/four_b.py (trycua/cua, libs/cua-s1). The model is a LoRA on
// Qwen/Qwen3.5-4B, merged and exported to ONNX. It is shown one screen state and a closed list of (element, action)
// options, one letter each, and picks the single best next action: one forward pass, then a softmax over the option
// letters' logits at the final position. Nothing is generated.
//
// The graph returns hidden states; Qwen3.5-4B ties its output layer to the embeddings, so the letters' logits are
// the final hidden state times the letters' embedding rows (head.safetensors, fp32).
//
// The multimodal adapter is a separate bundle: a screenshot instead of the accessibility tree, a vision graph that
// turns it into one embedding per 2 x 2 patch block, and a decoder that splices those in at the <|image_pad|> tokens.

import { Tokenizer } from "@huggingface/tokenizers";
import type { InferenceSession, Tensor } from "onnxruntime-common";
import { dropOtherRevisions, fetchFile, parseSafetensors, pool, type Progress } from "./fetch.ts";
import { preprocess, ropePositions, visionInputs, type ImageLike, type VisionConfig } from "./four-b-vision.ts";

export * from "./four-b-vision.ts";
import type { OrtModule } from "./model.ts";

export const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/** One candidate (element, action) decision. `entityId` names the source-record value a `fill` would enter. */
export interface FourBOption {
  elementId: string;
  role: string;
  label: string;
  action: string;
  entityId?: string | null;
}

export interface FourBContext {
  /** which app or page the screen belongs to (cua-bench-s1 `CuaTask.app`) */
  app: string;
  /** e.g. "form_filling", "login_auth", "consent_checkbox" (cua-bench-s1 `FAMILIES`) */
  taskFamily: string;
  /** the screen as an accessibility tree (text bundle); renderAxTree writes the format the model was trained on */
  axTree?: string;
  /** the screen as RGBA pixels (multimodal bundle), e.g. from canvas getImageData */
  screenshot?: ImageLike;
  /** the user's objective, when the state does not already show it */
  goal?: string;
}

export const SYSTEM_PROMPT =
  "You are a one-pass computer-use decision model. You are shown the current state of a screen and a fixed, closed "
  + "list of candidate (element, action) options, each given a single letter. Choose exactly one option: the single "
  + "best next action to take. Answer with ONLY that option's letter -- no words, no punctuation, no explanation.";

/** Caller-supplied text can never become a chat control token: `<|name|>` becomes `<¦name¦>`. (Upstream passes it
 * through; a web page's labels are not trusted input.) */
const inert = (s: string) => s.replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>");

export function describeOption(letter: string, o: FourBOption): string {
  const action = o.action === "fill" && o.entityId ? `${o.action} (with entity '${o.entityId}')` : o.action;
  return `${letter}. ${o.role} "${o.label}" -> ${action}`;
}

export type Modality = "text" | "multimodal";

/** four_b.build_prompt's user message text. */
export function buildPrompt(options: FourBOption[], c: FourBContext, modality: Modality = "text"): string {
  if (!options.length) throw new RangeError("no options to choose from");
  if (options.length > LETTERS.length) throw new RangeError(`${options.length} options exceeds the ${LETTERS.length}-letter budget`);
  if (modality === "text" && !c.axTree) throw new RangeError("the text modality requires an accessibility tree");
  const lines = options.map((o, i) => describeOption(LETTERS[i], o)).join("\n");
  return (c.goal ? `Goal: ${c.goal}\n\n` : "")
    + `App: ${c.app}\nTask family: ${c.taskFamily}\n\n`
    + (modality === "text" ? `Accessibility tree:\n${c.axTree}\n\n` : "The current screenshot is attached.\n\n")
    + `Options:\n${lines}\n\nAnswer with a single letter.`;
}

/** Python str.strip(), which Jinja's `trim` is: the characters str.isspace() accepts, not quite JavaScript's trim(). */
const PY_SPACE = new Set(String.fromCodePoint(9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004,
  0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000));
function pyStrip(s: string): string {
  let a = 0, b = s.length;
  while (a < b && PY_SPACE.has(s[a])) a++;
  while (b > a && PY_SPACE.has(s[b - 1])) b--;
  return s.slice(a, b);
}

/** Qwen3.5's chat template for [system, user] with add_generation_prompt (thinking left on, as upstream calls it).
 * Multimodal: the image block comes first, as one <|image_pad|> the caller expands to the image's token count. */
export function renderChat(options: FourBOption[], c: FourBContext, modality: Modality = "text"): string {
  const user = buildPrompt(
    options.map((o) => ({ ...o, role: inert(o.role), label: inert(o.label), action: inert(o.action), entityId: o.entityId == null ? o.entityId : inert(o.entityId) })),
    { app: inert(c.app), taskFamily: inert(c.taskFamily), axTree: c.axTree == null ? c.axTree : inert(c.axTree), goal: c.goal == null ? c.goal : inert(c.goal) },
    modality,
  );
  const image = modality === "multimodal" ? "<|vision_start|><|image_pad|><|vision_end|>" : "";
  return `<|im_start|>system\n${pyStrip(SYSTEM_PROMPT)}<|im_end|>\n<|im_start|>user\n${pyStrip(image + user)}<|im_end|>\n<|im_start|>assistant\n<think>\n`;
}

/** A row of the screen, in display order (cua-bench-s1 datagen/render.py). */
export interface AxRow { id: string; role: string; label: string; value?: string; checked?: boolean; required?: boolean }
/** A source-record value a `fill` option points to by `id`. */
export interface AxEntity { id: string; label: string; value: string }

const pyBool = (b: unknown) => (b ? "True" : "False");

/** The synthetic accessibility tree cua-bench-s1 trains and evaluates on (render.render_ax_tree): title, the goal and
 * the source record the `fill (with entity '...')` options point into, then one line per element. */
export function renderAxTree(title: string, rows: AxRow[], entities?: AxEntity[] | null, goal?: string | null): string {
  const lines = [`# ${title}`, ""];
  const src: string[] = [];
  if (goal) {
    src.push("This is ONE turn. Judge every element against the screen's CURRENT state as shown below, NOT against the "
      + "state it would be in after your other choices this turn. So: do not submit or advance while ANY field on this "
      + "screen is still empty and has a value available in the source record, or a required box is still unticked -- "
      + "even if you are also choosing to fill or tick it in this same turn; advancing comes on a later turn. An empty "
      + "field the record has no value for is not fillable and never blocks advancing. Only fill a field when the record "
      + "holds a value that genuinely belongs in THAT field: never repurpose a value that belongs to a different field, a "
      + "different person, or a different point in time.");
    src.push(`Goal: ${goal}`);
  }
  if (entities?.length) {
    src.push("Source record:");
    for (const e of entities) src.push(`  [${e.id}] ${e.label}: ${e.value}`);
  }
  if (src.length) lines.push(...src, "");
  for (const r of rows) {
    if (r.role === "Edit" || r.role === "Select") lines.push(`- [${r.id}] ${r.role} "${r.label}" value="${r.value ?? ""}" required=${pyBool(r.required)}`);
    else if (r.role === "CheckBox") lines.push(`- [${r.id}] ${r.role} "${r.label}" checked=${pyBool(r.checked)} required=${pyBool(r.required)}`);
    else lines.push(`- [${r.id}] ${r.role} "${r.label}"`);
  }
  return lines.join("\n");
}

interface IOInfo { name: string; type: "float16" | "float32" | "int64"; shape: (number | string)[]; empty?: number[] }

export interface FourBVariant {
  model: string;
  data: string[];
  bytes: number;
  io_dtype: "float32";
  sizes: Record<string, number>;
  inputs: IOInfo[];
  outputs: IOInfo[];
  /** against upstream's FourBModel (fp32 PyTorch) on the bundled fixtures; `quality` is how often this variant is right */
  parity?: { max_abs_dp: number; argmax_flips: number; tasks: number; quality: string };
}

export interface FourBManifest {
  name: string;
  /** adapter repo@commit */
  run: string;
  modality: Modality;
  /** base model repo@commit */
  base: string;
  hidden_size: number;
  letters: string;
  letter_ids: number[];
  files: { head: string; tokenizer: string; tokenizer_config: string };
  variants: Record<string, FourBVariant>;
  /** multimodal bundles: the vision graph (patches -> one embedding per 2 x 2 block) and its preprocessing */
  vision?: { model: string; data: string[]; bytes: number; sizes: Record<string, number>; config: VisionConfig; image_token_id: number; parity?: FourBVariant["parity"] };
}

export interface ScoredOption { letter: string; option: FourBOption; probability: number }

export interface FourBResult {
  /** one per option, in the given order */
  options: ScoredOption[];
  /** the highest-probability option */
  best: ScoredOption;
  tokens: number;
  latencyMs: number;
}

/** The likeliest action for each element, as cua-bench-s1 scores the model: every element's options are compared
 * among themselves. A screen is handled right when each element's choice is the expected one (mostly `skip`). */
export function elementDecisions(options: ScoredOption[]): Map<string, ScoredOption> {
  const best = new Map<string, ScoredOption>();
  for (const o of options) {
    const b = best.get(o.option.elementId);
    if (!b || o.probability > b.probability) best.set(o.option.elementId, o);
  }
  return best;
}

export class CuaS1FourB {
  readonly manifest: FourBManifest;
  readonly variant: string;
  readonly tokenizer: Tokenizer;
  private ort: OrtModule;
  private session: InferenceSession;
  private head: Float32Array;
  private v: FourBVariant;
  private vision: InferenceSession | null;
  private queue: Promise<unknown> = Promise.resolve();

  constructor(a: { ort: OrtModule; session: InferenceSession; head: Float32Array; tokenizer: Tokenizer; manifest: FourBManifest; variant: string; vision?: InferenceSession }) {
    this.ort = a.ort; this.session = a.session; this.head = a.head; this.tokenizer = a.tokenizer; this.vision = a.vision ?? null;
    this.manifest = a.manifest; this.variant = a.variant; this.v = a.manifest.variants[a.variant];
    if (!this.v) throw new Error(`unknown variant ${a.variant}; have ${Object.keys(a.manifest.variants)}`);
    if (a.manifest.modality === "multimodal" && (!this.vision || !a.manifest.vision)) throw new Error("a multimodal bundle needs its vision graph");
  }

  get modality(): Modality { return this.manifest.modality; }

  /** Token ids of the rendered chat; for a screenshot, <|image_pad|> is expanded to `imageTokens` copies. */
  encode(options: FourBOption[], c: FourBContext, imageTokens = 0): number[] {
    const ids = this.tokenizer.encode(renderChat(options, c, this.modality), { add_special_tokens: false }).ids;
    if (this.modality === "text") return ids;
    const pad = this.manifest.vision!.image_token_id, at = ids.indexOf(pad);
    if (at < 0 || ids.indexOf(pad, at + 1) >= 0) throw new Error("expected exactly one image placeholder");
    return [...ids.slice(0, at), ...new Array<number>(imageTokens).fill(pad), ...ids.slice(at + 1)];
  }

  /** The screenshot through the vision graph: one row of image_embeds per 2 x 2 patch block. */
  async imageEmbeds(img: ImageLike): Promise<{ embeds: Float32Array; gridH: number; gridW: number }> {
    const cfg = this.manifest.vision!.config, T = this.ort.Tensor;
    const p = preprocess(img, cfg), vi = visionInputs(p.gridH, p.gridW, cfg), P = p.gridH * p.gridW, hd = cfg.hidden_size / cfg.num_heads;
    const out = await this.vision!.run({
      patches: new T("float32", p.data, [P, p.data.length / P]),
      pos_idx: new T("int64", vi.posIdx, [P, 4]), pos_w: new T("float32", vi.posW, [P, 4]),
      cos: new T("float32", vi.cos, [P, hd]), sin: new T("float32", vi.sin, [P, hd]),
    }, ["image_embeds"]);
    return { embeds: out.image_embeds.data as Float32Array, gridH: p.gridH, gridW: p.gridW };
  }

  /** Letter probabilities for token ids that end at the answer position (a rendered chat). With an image, `pos` holds
   * the mRoPE positions and `image` the embeddings for its <|image_pad|> tokens. */
  probsForIds(ids: number[], n: number, image?: { embeds: Float32Array; pos: number[][] }): Promise<number[]> {
    const run = this.queue.then(async () => {
      const S = ids.length, T = this.ort.Tensor, d = this.manifest.hidden_size;
      const pos = image?.pos ?? [[...ids.keys()], [...ids.keys()], [...ids.keys()]];   // text: the three mRoPE rows are equal
      const feeds: Record<string, Tensor> = {
        input_ids: new T("int64", BigInt64Array.from(ids, BigInt), [1, S]),
        attention_mask: new T("int64", new BigInt64Array(S).fill(1n), [1, S]),
        position_ids: new T("int64", BigInt64Array.from(pos.flat(), BigInt), [3, 1, S]),
      };
      if (this.v.inputs.some((i) => i.name === "image_embeds")) {
        const e = image?.embeds ?? new Float32Array(d);   // no image: one row of zeros, never gathered
        feeds.image_embeds = new T("float32", e, [e.length / d, d]);
      }
      for (const i of this.v.inputs) if (i.empty) feeds[i.name] = new T("float32", new Float32Array(i.empty.reduce((a, b) => a * b, 1)), i.empty);
      const out = await this.session.run(feeds, ["hidden_states"]);
      const h = out.hidden_states.data as Float32Array;
      const last = h.subarray((S - 1) * d, S * d);
      const z = Array.from({ length: n }, (_, k) => {
        let s = 0;
        for (let j = 0, row = k * d; j < d; j++) s += this.head[row + j] * last[j];
        return s;
      });
      const m = Math.max(...z), e = z.map((x) => Math.exp(x - m)), sum = e.reduce((a, b) => a + b, 0);
      return e.map((x) => x / sum);
    });
    this.queue = run.catch(() => undefined);
    return run;
  }

  /** four_b.FourBModel.forward: a probability per option, in the given order. */
  async score(options: FourBOption[], c: FourBContext): Promise<FourBResult> {
    const t0 = performance.now();
    let ids: number[], p: number[];
    if (this.modality === "multimodal") {
      if (!c.screenshot) throw new RangeError("the multimodal bundle needs a screenshot");
      const img = await this.imageEmbeds(c.screenshot);
      ids = this.encode(options, c, img.embeds.length / this.manifest.hidden_size);
      p = await this.probsForIds(ids, options.length, { embeds: img.embeds, pos: ropePositions(ids, img.gridH, img.gridW, this.manifest.vision!.image_token_id, this.manifest.vision!.config.merge_size) });
    } else {
      ids = this.encode(options, c);
      p = await this.probsForIds(ids, options.length);
    }
    const scored = options.map((option, i) => ({ letter: LETTERS[i], option, probability: p[i] }));
    return { options: scored, best: scored.reduce((a, b) => (b.probability > a.probability ? b : a)), tokens: ids.length, latencyMs: performance.now() - t0 };
  }

  async release() { await this.session.release(); await this.vision?.release(); }
}

export interface LoadFourBOptions {
  ort: OrtModule;
  /** variant in manifest.variants; default: the first one listed */
  variant?: string;
  /** default ["webgpu", "wasm"] */
  executionProviders?: InferenceSession.SessionOptions["executionProviders"];
  onProgress?: (p: Progress) => void;
  /** Cache Storage bucket for the weights; null disables caching. Default "cua-s1-4b-v1". */
  cacheName?: string | null;
  /** files fetched at once. Default 2: on a bandwidth-limited link more streams are slower in aggregate. */
  concurrency?: number;
  sessionOptions?: InferenceSession.SessionOptions;
}

const join = (base: string, path: string) => `${base.replace(/\/$/, "")}/${path}`;

/** Load a packaged cua-s1-4b model (export/four_b/package.py output) from a URL. */
export async function loadCuaS1FourB(baseUrl: string, o: LoadFourBOptions): Promise<CuaS1FourB> {
  const decode = (b: Uint8Array) => new TextDecoder().decode(b);
  const manifest = JSON.parse(decode(await fetchFile(join(baseUrl, "manifest.json"), { cacheName: null }))) as FourBManifest;   // always fresh: it names the revision
  const variant = o.variant ?? Object.keys(manifest.variants)[0];
  const v = manifest.variants[variant];
  if (!v) throw new Error(`unknown variant ${variant}; have ${Object.keys(manifest.variants).join(", ")}`);
  const cacheName = o.cacheName === undefined ? "cua-s1-4b-v1" : o.cacheName;
  for (const [p, bytes] of Object.entries(v.sizes)) o.onProgress?.({ file: p, loaded: 0, total: bytes });   // the total is known up front
  const get = (p: string) => fetchFile(join(baseUrl, p), { cacheName, onProgress: o.onProgress, file: p, bytes: v.sizes[p], rev: manifest.run });
  const [tokJson, tokCfg] = (await Promise.all([get(manifest.files.tokenizer), get(manifest.files.tokenizer_config)])).map(decode);
  const vis = manifest.vision;
  if (vis) for (const [p, bytes] of Object.entries(vis.sizes)) o.onProgress?.({ file: p, loaded: 0, total: bytes });
  const visFiles = vis ? [vis.model, ...vis.data] : [];
  const getAny = (p: string) => fetchFile(join(baseUrl, p), { cacheName, onProgress: o.onProgress, file: p, bytes: v.sizes[p] ?? vis?.sizes[p], rev: manifest.run });
  const [head, graph, ...rest] = await pool([manifest.files.head, v.model, ...v.data, ...visFiles].map((p) => () => getAny(p)), o.concurrency ?? 2);
  const data = rest.slice(0, v.data.length), visData = rest.slice(v.data.length);
  const weight = parseSafetensors(head.slice().buffer).weight;
  if (!weight || weight.shape[0] !== manifest.letters.length || weight.shape[1] !== manifest.hidden_size) throw new Error("head.safetensors: expected weight [letters, hidden]");
  const session = await o.ort.InferenceSession.create(graph, {
    graphOptimizationLevel: "all",
    executionProviders: o.executionProviders ?? ["webgpu", "wasm"],
    externalData: v.data.map((p, i) => ({ path: p.split("/").pop()!, data: data[i] })),
    ...o.sessionOptions,
  });
  const vision = vis ? await o.ort.InferenceSession.create(visData[0], {
    graphOptimizationLevel: "all",
    executionProviders: o.executionProviders ?? ["webgpu", "wasm"],
    externalData: vis.data.map((p, i) => ({ path: p.split("/").pop()!, data: visData[i + 1] })),
    ...o.sessionOptions,
  }) : undefined;
  await dropOtherRevisions(baseUrl, manifest.run, cacheName);
  return new CuaS1FourB({ ort: o.ort, session, head: weight.data, tokenizer: new Tokenizer(JSON.parse(tokJson), JSON.parse(tokCfg)), manifest, variant, vision });
}
