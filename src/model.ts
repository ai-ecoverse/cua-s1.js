// cua-s1 scorer over onnxruntime (web or node). Every element of a form is scored in one batch: the context is the
// element (renderContext), the options are one `fill` per document entity plus check / click / skip. Every element
// shares those options, so the shared-options graph encodes them once per plan instead of once per element.

import type { InferenceSession, Tensor } from "onnxruntime-common";
import { collate, collateShared, type Example } from "./collate.ts";
import { decode, filterElements, normalizeTitle, orderDecisions, renderContext, renderOptions, type Decision, type Element, type Entity } from "./schema.ts";

/** The subset of onnxruntime-web / onnxruntime-node the runtime uses. */
export interface OrtModule {
  InferenceSession: { create(model: Uint8Array | string, options?: InferenceSession.SessionOptions): Promise<InferenceSession> };
  Tensor: new (type: Tensor.Type, data: Tensor.DataType, dims: readonly number[]) => Tensor;
}

export interface Manifest {
  name: string;
  /** Hugging Face repo@commit the weights came from */
  source: string;
  model: string;
  sha256: string;
  bytes: number;
  context_tokens: number;
  option_tokens: number;
  inputs: string[];
  outputs: string[];
  /** onnxruntime vs PyTorch on synthetic episodes at export time */
  parity: Parity;
  /** the same model for batches that share one option list: options encoded once, not once per row */
  shared_options?: { model: string; sha256: string; bytes: number; inputs: string[]; parity: Parity };
}

export interface Parity { decisions: number; max_abs_dp: number; argmax_flips: number; top1_vs_labels: number }

export interface PlanOptions {
  /** decisions below this probability are dropped from the ordered plan. Default 0.5 */
  minConfidence?: number;
  /** keep one click on a control labelled exactly "Submit" or "Submit Form". Default false */
  allowSubmit?: boolean;
}

export interface Plan {
  /** one decision per actionable element, in element order */
  decisions: Decision[];
  /** what to execute, in order: fills, checkboxes, then at most one submit click */
  actions: Decision[];
  latencyMs: number;
}

export class CuaS1 {
  readonly manifest: Manifest;
  /** the session runs manifest.shared_options.model rather than manifest.model */
  readonly sharedOptions: boolean;
  private ort: OrtModule;
  private session: InferenceSession;

  constructor(a: { ort: OrtModule; session: InferenceSession; manifest: Manifest; sharedOptions?: boolean }) {
    this.ort = a.ort; this.session = a.session; this.manifest = a.manifest; this.sharedOptions = a.sharedOptions ?? false;
  }

  /** Probabilities for each example: one row per example, one value per option. */
  async score(examples: Example[]): Promise<number[][]> {
    if (!this.sharedOptions) return this.scoreRows(examples);
    const groups = new Map<string, number[]>();   // one run per distinct option list; a plan has exactly one
    examples.forEach((e, i) => { const k = JSON.stringify(e.options); (groups.get(k) ?? groups.set(k, []).get(k)!).push(i); });
    const out = new Array<number[]>(examples.length);
    for (const rows of groups.values()) {
      const probs = await this.scoreShared(rows.map((i) => examples[i].context), examples[rows[0]].options);
      rows.forEach((i, j) => { out[i] = probs[j]; });
    }
    return out;
  }

  private async scoreShared(contexts: string[], options: string[]): Promise<number[][]> {
    const m = this.manifest;
    const b = collateShared(contexts, options, m.context_tokens, m.option_tokens);
    const T = this.ort.Tensor;
    const out = await this.session.run({
      context_ids: new T("int64", b.contextIds, [b.batch, b.contextLen]),
      context_mask: new T("bool", b.contextMask, [b.batch, b.contextLen]),
      option_ids: new T("int64", b.optionIds, [b.options, b.optionLen]),
      option_token_mask: new T("bool", b.optionTokenMask, [b.options, b.optionLen]),
    }, ["probabilities"]);
    const p = out.probabilities.data as Float32Array;
    return contexts.map((_, i) => Array.from(p.subarray(i * b.options, (i + 1) * b.options)));
  }

  private async scoreRows(examples: Example[]): Promise<number[][]> {
    const m = this.manifest;
    const b = collate(examples, m.context_tokens, m.option_tokens);
    const T = this.ort.Tensor;
    const out = await this.session.run({
      context_ids: new T("int64", b.contextIds, [b.batch, b.contextLen]),
      context_mask: new T("bool", b.contextMask, [b.batch, b.contextLen]),
      option_ids: new T("int64", b.optionIds, [b.batch, b.options, b.optionLen]),
      option_token_mask: new T("bool", b.optionTokenMask, [b.batch, b.options, b.optionLen]),
      option_mask: new T("bool", b.optionMask, [b.batch, b.options]),
    }, ["probabilities"]);
    const p = out.probabilities.data as Float32Array;
    return b.counts.map((k, i) => Array.from(p.subarray(i * b.options, i * b.options + k)));
  }

  /**
   * Decide every actionable element of a form from the document's entities, then order what to execute. Nothing is
   * executed here: the caller applies `actions` (and should show them first).
   */
  async plan(formTitle: string, elements: Element[], entities: Entity[], o: PlanOptions = {}): Promise<Plan> {
    const title = normalizeTitle(formTitle);
    const actionable = filterElements(elements).map((e, i) => ({ ...e, index: e.index ?? i }));
    const t0 = performance.now();
    const options = renderOptions(entities);
    const probs = actionable.length ? await this.score(actionable.map((e) => ({ context: renderContext(title, e), options }))) : [];
    const decisions = actionable.map((element, i) => {
      const p = probs[i];
      const best = p.reduce((m, v, j) => (v > p[m] ? j : m), 0);
      return { element, ...decode(best, entities), probability: p[best], distribution: p };
    });
    return { decisions, actions: orderDecisions(decisions, o.minConfidence ?? 0.5, { allowSubmit: o.allowSubmit }), latencyMs: performance.now() - t0 };
  }

  async release() { await this.session.release(); }
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes as BufferSource));
  return Array.from(d, (x) => x.toString(16).padStart(2, "0")).join("");
}

/**
 * Load a packaged model (export.py output) from a URL; the graph's SHA-256 is checked against the manifest. The
 * shared-options graph is used when the manifest has one, unless `sharedOptions: false` asks for the per-row graph.
 */
export async function loadCuaS1(baseUrl: string, o: { ort: OrtModule; sessionOptions?: InferenceSession.SessionOptions; sharedOptions?: boolean }): Promise<CuaS1> {
  const base = baseUrl.replace(/\/$/, "");
  const get = async (p: string) => { const r = await fetch(`${base}/${p}`); if (!r.ok) throw new Error(`${base}/${p}: HTTP ${r.status}`); return r; };
  const manifest = (await (await get("manifest.json")).json()) as Manifest;
  const shared = o.sharedOptions !== false && manifest.shared_options ? manifest.shared_options : null;
  const file = shared ?? manifest;
  const graph = new Uint8Array(await (await get(file.model)).arrayBuffer());
  const digest = await sha256(graph);
  if (digest !== file.sha256) throw new Error(`${file.model}: SHA-256 ${digest} does not match the manifest`);
  const session = await o.ort.InferenceSession.create(graph, { executionProviders: ["wasm"], graphOptimizationLevel: "all", ...o.sessionOptions });
  return new CuaS1({ ort: o.ort, session, manifest, sharedOptions: !!shared });
}
