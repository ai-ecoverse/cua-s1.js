// Fetching multi-gigabyte model bundles (from kev.js's src/load.ts): files are kept in Cache Storage keyed by the
// model revision, streamed with progress, checked against their expected size, and fetched a few at a time.

export interface Progress { file: string; loaded: number; total: number }

async function readWithProgress(res: Response, file: string, onProgress?: (p: Progress) => void, expected = 0): Promise<Uint8Array> {
  const total = expected || Number(res.headers.get("content-length") ?? 0);
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.length;
    onProgress({ file, loaded, total: total || loaded });
  }
  if (chunks.length === 1) return chunks[0];
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/** The Cache Storage key for one file of one model revision: a new checkpoint never reads an old one's bytes. */
const cacheKey = (url: string, rev?: string) => (rev ? `${url}${url.includes("?") ? "&" : "?"}rev=${encodeURIComponent(rev)}` : url);

export interface FetchOptions { cacheName?: string | null; onProgress?: (p: Progress) => void; file?: string; bytes?: number; rev?: string }

export async function fetchFile(url: string, o: FetchOptions = {}): Promise<Uint8Array> {
  const file = o.file ?? url;
  const cache = o.cacheName && typeof caches !== "undefined" ? await caches.open(o.cacheName) : null;
  const key = cacheKey(url, o.rev);
  const hit = await cache?.match(key);
  if (hit) {
    const data = await readWithProgress(hit, file, o.onProgress, o.bytes);
    if (!o.bytes || data.length === o.bytes) return data;
    await cache!.delete(key);   // truncated entry (a reload mid-download): fetch it again
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  // read first and cache afterwards: cache.put(res.clone()) resolves only once the whole body is in, so a large file
  // would report no progress at all
  const data = await readWithProgress(res, file, o.onProgress, o.bytes);
  if (o.bytes && data.length !== o.bytes) throw new Error(`${file}: expected ${o.bytes} bytes, received ${data.length}`);
  if (cache) {
    try {
      await cache.put(key, new Response(data as BodyInit, { headers: { "content-type": res.headers.get("content-type") ?? "application/octet-stream", "content-length": String(data.length) } }));
    } catch { /* quota: run uncached */ }
  }
  return data;
}

/** Delete this model's cached files from other revisions: a superseded checkpoint is gigabytes of quota. */
export async function dropOtherRevisions(baseUrl: string, rev: string, cacheName?: string | null) {
  if (!cacheName || typeof caches === "undefined") return;
  const cache = await caches.open(cacheName);
  const prefix = `${baseUrl.replace(/\/$/, "")}/`;
  for (const req of await cache.keys()) {
    if (req.url.startsWith(prefix) && new URL(req.url).searchParams.get("rev") !== rev) await cache.delete(req);
  }
}

/** Run jobs with a bounded number in flight: hundreds of parallel shard requests are slower than a handful. */
export async function pool<T>(jobs: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const out = new Array<T>(jobs.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, jobs.length) }, async () => {
    for (let i = next++; i < jobs.length; i = next++) out[i] = await jobs[i]();
  }));
  return out;
}

/** F32 tensors from a safetensors file. */
export function parseSafetensors(buf: ArrayBuffer): Record<string, { shape: number[]; data: Float32Array }> {
  const n = Number(new DataView(buf).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n))) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const out: Record<string, { shape: number[]; data: Float32Array }> = {};
  for (const [name, t] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (t.dtype !== "F32") throw new Error(`${name}: expected F32, got ${t.dtype}`);
    const [a, b] = t.data_offsets;
    out[name] = { shape: t.shape, data: new Float32Array(buf.slice(8 + n + a, 8 + n + b)) };   // copy: offsets need not be 4-aligned
  }
  return out;
}
