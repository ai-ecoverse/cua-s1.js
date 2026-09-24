import * as ort from "onnxruntime-web/webgpu";
import wasm from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import mjs from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { elementDecisions, loadCuaS1FourB, LETTERS, type CuaS1FourB, type FourBOption, type ScoredOption } from "../src/four-b.ts";
import type { OrtModule } from "../src/model.ts";
import textFixtures from "../fixtures/cua-s1-4b-0.2.json";
import mmFixtures from "../fixtures/cua-s1-4b-0.2-multimodal.json";

const shots = import.meta.glob("../fixtures/cua-s1-4b-0.2-multimodal/*.png", { query: "?url", import: "default", eager: true }) as Record<string, string>;

ort.env.wasm.wasmPaths = { wasm, mjs };
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;

// Weights live on Hugging Face for the published page; the dev server serves public/models/. ?models=<url> overrides.
const HF_BASE = "https://huggingface.co/ai-ecoverse/cua-s1.js/resolve/main";
const MODEL_BASE = new URLSearchParams(location.search).get("models") ?? (import.meta.env.VITE_MODEL_BASE as string | undefined) ?? (import.meta.env.DEV ? "models" : HF_BASE);

interface Task { id: string; family: string; app: string; goal: string | null; ax_tree: string | null; screenshot?: string; gold: string[]; options: { element_id: string; role: string; label: string; action: string; entity_id: string | null }[] }
const modality = new URLSearchParams(location.search).get("modality") === "multimodal" ? "multimodal" : "text";
const bundle = modality === "multimodal" ? "cua-s1-4b-0.2-multimodal" : "cua-s1-4b-0.2";
const tasks = ((modality === "multimodal" ? mmFixtures : textFixtures) as { fixtures: Task[] }).fixtures;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const status = (s: string) => { $("status").textContent = s; };
const select = $<HTMLSelectElement>("task"), tree = $<HTMLTextAreaElement>("tree"), shot = $<HTMLImageElement>("shot"), tbody = $("options").querySelector("tbody")!;
let model: CuaS1FourB | null = null;
$(`mode-${modality}`).classList.add("current");
tree.hidden = modality === "multimodal"; shot.hidden = modality === "text";
for (const a of document.querySelectorAll<HTMLAnchorElement>("#mode-text, #mode-multimodal")) {   // keep ?models= when switching
  const u = new URL(location.href); u.searchParams.set("modality", a.id.slice(5)); a.href = u.search;
}

/** RGBA pixels of the task's screenshot, as the model reads them. */
async function pixels(t: Task) {
  const bmp = await createImageBitmap(await (await fetch(shots[`../fixtures/${t.screenshot}`])).blob());
  const c = new OffscreenCanvas(bmp.width, bmp.height), g = c.getContext("2d")!;
  g.drawImage(bmp, 0, 0);
  return g.getImageData(0, 0, bmp.width, bmp.height);
}

// gui360_<app>_1_<episode>_<step>
tasks.forEach((t, i) => select.append(new Option(`${t.family} · ${t.id.replace(/^gui360_/, "").replace(/_(\d+)$/, ", step $1")}`, String(i))));
const task = () => tasks[Number(select.value)];
const options = (t: Task): FourBOption[] => t.options.map((o) => ({ elementId: o.element_id, role: o.role, label: o.label, action: o.action, entityId: o.entity_id }));
/** The answer key's action: every other element's gold is skip, so a ✓ there would mark nearly every row. */
const action = (t: Task) => t.gold.filter((l) => t.options[LETTERS.indexOf(l)].action !== "skip");

/** The options the model would act on: on each element, the likeliest of its actions, unless that is skip. */
const acts = (scored: ScoredOption[]) => [...elementDecisions(scored).values()].filter((o) => o.option.action !== "skip").map((o) => o.letter);

function show(probs?: number[]) {
  const t = task();
  const chosen = probs ? acts(options(t).map((option, i) => ({ letter: LETTERS[i], option, probability: probs[i] }))) : [];
  tbody.innerHTML = "";
  options(t).forEach((o, i) => {
    const letter = LETTERS[i];
    const tr = document.createElement("tr");
    if (chosen.includes(letter)) tr.className = "best";
    if (probs) tr.dataset.p = String(probs[i]);
    const act = o.action === "fill" && o.entityId ? `fill ← ${o.entityId}` : o.action;
    const cells = [letter, `${o.role} “${o.label}” → ${act}`, action(t).includes(letter) ? "✓" : "", probs ? probs[i].toFixed(3) : ""];
    cells.forEach((c, k) => { const td = document.createElement("td"); td.textContent = c; if (k === 2) td.className = "gold"; if (k === 3) td.className = "p"; tr.append(td); });
    if (probs) { const td = document.createElement("td"); td.className = "bar"; td.append(Object.assign(document.createElement("span"), { style: `width:${(probs[i] * 100).toFixed(1)}%` })); tr.append(td); }
    tbody.append(tr);
  });
}

function pick() {
  const t = task();
  if (t.screenshot) shot.src = shots[`../fixtures/${t.screenshot}`];
  else tree.value = t.ax_tree ?? "";
  $("goal").textContent = t.goal ?? "";
  $("task-meta").textContent = `${t.options.length} options`;
  $("score-meta").textContent = "";
  show();
}
select.onchange = () => { pick(); if (model) void score(); };
pick();

async function score() {
  if (!model) return;
  const t = task();
  $<HTMLButtonElement>("score").disabled = true;
  try {
    const goal = t.goal ?? undefined;
    const r = await model.score(options(t), modality === "multimodal"
      ? { app: t.app, taskFamily: t.family, goal, screenshot: await pixels(t) }
      : { app: t.app, taskFamily: t.family, goal, axTree: tree.value });
    show(r.options.map((o) => o.probability));
    const want = action(t), did = acts(r.options);
    const same = did.length === want.length && did.every((l) => want.includes(l));
    $("score-meta").textContent = `${r.tokens} tokens in ${r.latencyMs.toFixed(0)} ms · `
      + (did.length ? `would act on ${did.join(", ")}` : "would skip everything")
      + (same ? " ✓" : ` (recorded: ${want.join(", ") || "nothing"})`);
  } finally { $<HTMLButtonElement>("score").disabled = false; }
}
$("score").onclick = () => void score();

const hasGpu = "gpu" in navigator && !!(await (navigator as Navigator & { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter().catch(() => null));
const t0 = performance.now();
const got = new Map<string, [number, number]>();
model = await loadCuaS1FourB(new URL(`${MODEL_BASE.replace(/\/$/, "")}/${bundle}`, location.href).href, {
  ort: ort as unknown as OrtModule,
  executionProviders: hasGpu ? ["webgpu"] : ["wasm"],
  onProgress: (p) => {
    got.set(p.file, [p.loaded, p.total]);
    let a = 0, b = 0; for (const [l, t] of got.values()) { a += l; b += t; }
    status(`Downloading ${(a / 1e9).toFixed(2)} of ${(b / 1e9).toFixed(2)} GB…`);
  },
});
status(`${bundle} loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${model.manifest.run.split("@")[0]}@${model.manifest.run.split("@")[1].slice(0, 7)} · ${model.variant} · ${hasGpu ? "WebGPU" : "WASM (no WebGPU: slow)"}`);
$<HTMLButtonElement>("score").disabled = false;
await score();
