// UTF-8 byte tensors in exactly the layout cua_s1.model.ByteCollator produces: byte + 1 (0 is padding), truncated
// by bytes, padded to the batch maxima, with padding masks.

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;
const encoder = new TextEncoder();

/** Python text.encode("utf-8", errors="replace")[:length], each byte + 1. A lone surrogate becomes "?" as in
 * Python; TextEncoder would emit U+FFFD instead. */
export function byteIds(text: string, length: number): number[] {
  const bytes = encoder.encode(text.replace(LONE_SURROGATE, "?"));
  const n = Math.min(bytes.length, length);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = bytes[i] + 1;
  return out;
}

export interface Example { context: string; options: string[] }

export interface Batch {
  batch: number;
  contextLen: number;
  options: number;
  optionLen: number;
  contextIds: BigInt64Array;
  contextMask: Uint8Array;
  optionIds: BigInt64Array;
  optionTokenMask: Uint8Array;
  optionMask: Uint8Array;
  /** options per example, to read the live probabilities back */
  counts: number[];
}

export function collate(examples: Example[], contextTokens: number, optionTokens: number): Batch {
  if (!examples.length) throw new RangeError("cannot collate an empty batch");
  const contexts = examples.map((e) => byteIds(e.context, contextTokens));
  const rows = examples.map((e) => e.options.map((o) => byteIds(o, optionTokens)));
  const B = examples.length;
  const L = Math.max(1, ...contexts.map((c) => c.length));
  const N = Math.max(...rows.map((r) => r.length));
  const T = Math.max(1, ...rows.flatMap((r) => r.map((t) => t.length)));
  const contextIds = new BigInt64Array(B * L), optionIds = new BigInt64Array(B * N * T);
  const contextMask = new Uint8Array(B * L), optionTokenMask = new Uint8Array(B * N * T), optionMask = new Uint8Array(B * N);
  contexts.forEach((c, b) => c.forEach((id, i) => { contextIds[b * L + i] = BigInt(id); contextMask[b * L + i] = 1; }));
  rows.forEach((r, b) => r.forEach((tokens, n) => {
    optionMask[b * N + n] = 1;
    tokens.forEach((id, t) => { const k = (b * N + n) * T + t; optionIds[k] = BigInt(id); optionTokenMask[k] = 1; });
  }));
  return { batch: B, contextLen: L, options: N, optionLen: T, contextIds, contextMask, optionIds, optionTokenMask, optionMask, counts: rows.map((r) => r.length) };
}
