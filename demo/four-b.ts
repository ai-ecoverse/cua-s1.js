import * as ort from "onnxruntime-web/webgpu";
import wasm from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import mjs from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { loadCuaS1FourB, type CuaS1FourB, type FourBOption } from "../src/four-b.ts";
import type { OrtModule } from "../src/model.ts";
import fixtures from "../fixtures/cua-s1-4b-0.1.json";

ort.env.wasm.wasmPaths = { wasm, mjs };
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;

// Weights live on Hugging Face for the published page; the dev server serves public/models/. ?models=<url> overrides.
const HF_BASE = "https://huggingface.co/ai-ecoverse/cua-s1.js/resolve/main";
const MODEL_BASE = new URLSearchParams(location.search).get("models") ?? (import.meta.env.VITE_MODEL_BASE as string | undefined) ?? (import.meta.env.DEV ? "models" : HF_BASE);

interface Task { id: string; family: string; app: string; ax_tree: string; gold: string[]; options: { element_id: string; role: string; label: string; action: string; entity_id: string | null }[] }
const tasks = (fixtures as { fixtures: Task[] }).fixtures;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const status = (s: string) => { $("status").textContent = s; };
const select = $<HTMLSelectElement>("task"), tree = $<HTMLTextAreaElement>("tree"), tbody = $("options").querySelector("tbody")!;
let model: CuaS1FourB | null = null;

tasks.forEach((t, i) => select.append(new Option(`${t.family} · ${t.app} · ${t.id.split("-").slice(-1)[0]}`, String(i))));
const task = () => tasks[Number(select.value)];
const options = (t: Task): FourBOption[] => t.options.map((o) => ({ elementId: o.element_id, role: o.role, label: o.label, action: o.action, entityId: o.entity_id }));

function show(probs?: number[]) {
  const t = task();
  const best = probs ? probs.indexOf(Math.max(...probs)) : -1;
  tbody.innerHTML = "";
  options(t).forEach((o, i) => {
    const letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i];
    const tr = document.createElement("tr");
    if (i === best) tr.className = "best";
    const act = o.action === "fill" && o.entityId ? `fill ← ${o.entityId}` : o.action;
    const cells = [letter, `${o.role} “${o.label}” → ${act}`, t.gold.includes(letter) ? "✓" : "", probs ? probs[i].toFixed(3) : ""];
    cells.forEach((c, k) => { const td = document.createElement("td"); td.textContent = c; if (k === 2) td.className = "gold"; if (k === 3) td.className = "p"; tr.append(td); });
    if (probs) { const td = document.createElement("td"); td.className = "bar"; td.append(Object.assign(document.createElement("span"), { style: `width:${(probs[i] * 100).toFixed(1)}%` })); tr.append(td); }
    tbody.append(tr);
  });
}

function pick() {
  const t = task();
  tree.value = t.ax_tree;
  $("task-meta").textContent = `${t.options.length} options, ${t.gold.length} accepted`;
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
    const r = await model.score(options(t), { app: t.app, taskFamily: t.family, axTree: tree.value });
    show(r.options.map((o) => o.probability));
    $("score-meta").textContent = `${r.tokens} tokens in ${r.latencyMs.toFixed(0)} ms · picked ${r.best.letter}${t.gold.includes(r.best.letter) ? " ✓" : " (not in the answer key)"}`;
  } finally { $<HTMLButtonElement>("score").disabled = false; }
}
$("score").onclick = () => void score();

const hasGpu = "gpu" in navigator && !!(await (navigator as Navigator & { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter().catch(() => null));
const t0 = performance.now();
const got = new Map<string, [number, number]>();
model = await loadCuaS1FourB(new URL(`${MODEL_BASE.replace(/\/$/, "")}/cua-s1-4b-0.1`, location.href).href, {
  ort: ort as unknown as OrtModule,
  executionProviders: hasGpu ? ["webgpu"] : ["wasm"],
  onProgress: (p) => {
    got.set(p.file, [p.loaded, p.total]);
    let a = 0, b = 0; for (const [l, t] of got.values()) { a += l; b += t; }
    status(`Downloading ${(a / 1e9).toFixed(2)} of ${(b / 1e9).toFixed(2)} GB…`);
  },
});
status(`cua-s1-4b-0.1 loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${model.manifest.run.split("@")[0]}@${model.manifest.run.split("@")[1].slice(0, 7)} · ${model.variant} · ${hasGpu ? "WebGPU" : "WASM (no WebGPU: slow)"}`);
$<HTMLButtonElement>("score").disabled = false;
await score();
