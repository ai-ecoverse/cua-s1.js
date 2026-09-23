import * as ort from "onnxruntime-web/wasm";
import wasm from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import mjs from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import { extractEntities, loadCuaS1, type CuaS1, type Decision, type Element, type Entity, type OrtModule, type Plan } from "../src/index.ts";

ort.env.wasm.wasmPaths = { wasm, mjs };
ort.env.wasm.numThreads = 1;   // a 3 MB model needs no threads, and GitHub Pages cannot send COOP/COEP headers

// Weights live on Hugging Face for the published page; the dev server serves public/models/. ?models=<url> overrides.
const HF_BASE = "https://huggingface.co/ai-ecoverse/cua-s1.js/resolve/main";
const MODEL_BASE = new URLSearchParams(location.search).get("models") ?? (import.meta.env.VITE_MODEL_BASE as string | undefined) ?? (import.meta.env.DEV ? "models" : HF_BASE);

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const form = $<HTMLFormElement>("target");
const doc = $<HTMLTextAreaElement>("doc");

doc.value = `Northwind Clinic — patient intake summary
Name: Amara Ivanova
First name: Amara
Last name: Ivanova
DOB: 03/14/1987
Tel: (503) 555-0142
Email: amara.ivanova@example.invalid
Street: 4881 Station Court
City: Portland
ZIP: 97205
Emergency contact: Dmitri Ivanov
Emergency contact phone: (503) 555-0199
Member ID: NWC-448-2291
Allergies: Latex, penicillin
Employer: Harbor Freight Logistics
Work phone: (503) 555-0110`;

const status = (s: string) => { $("status").textContent = s; };
let model: CuaS1 | null = null;
let last: { plan: Plan; entities: Entity[]; nodes: Map<string, HTMLElement> } | null = null;

const showEntities = () => { $("entities").textContent = `${extractEntities(doc.value).length} Label: value pairs`; };
doc.oninput = showEntities; showEntities();

/** Text of a label without the text of the control inside it. */
function labelText(control: HTMLElement): string {
  const label = control.closest("label") ?? (control.id ? document.querySelector<HTMLLabelElement>(`label[for="${control.id}"]`) : null);
  if (!label) return control.getAttribute("aria-label") ?? "";
  return Array.from(label.childNodes).filter((n) => n.nodeType === Node.TEXT_NODE).map((n) => n.textContent).join(" ").replace(/\s+/g, " ").trim();
}

/** The form as the model's element observations. Selects and radios are left out: cua-s1 was trained on edit
 * fields, checkboxes and buttons only. */
function observe(): { elements: Element[]; nodes: Map<string, HTMLElement> } {
  const nodes = new Map<string, HTMLElement>();
  const elements: Element[] = [];
  form.querySelectorAll<HTMLElement>("input, textarea, button").forEach((n, index) => {
    const token = `el-${index}`;
    if (n instanceof HTMLInputElement && n.type === "checkbox") elements.push({ role: "CheckBox", label: labelText(n), checked: n.checked, index, token });
    else if (n instanceof HTMLButtonElement) elements.push({ role: "Button", label: n.textContent?.trim() ?? "", index, token });
    else if (n instanceof HTMLInputElement || n instanceof HTMLTextAreaElement) {
      if (n instanceof HTMLInputElement && ["radio", "hidden", "file", "submit", "button"].includes(n.type)) return;
      elements.push({ role: "Edit", label: labelText(n), value: n.value, placeholder: n.placeholder, index, token });
    } else return;
    nodes.set(token, n);
  });
  return { elements, nodes };
}

function describe(d: Decision, entities: Entity[]): string {
  return d.action === "fill" ? `fill ← ${entities[d.entityIndex!].label}: ${entities[d.entityIndex!].value}` : d.action;
}

function render(plan: Plan, entities: Entity[]) {
  const acting = new Set(plan.actions);
  const tbody = $("decisions").querySelector("tbody")!;
  tbody.innerHTML = "";
  for (const d of plan.decisions) {
    const tr = document.createElement("tr");
    tr.className = acting.has(d) ? "act" : d.action === "skip" ? "skip" : "dropped";
    const second = d.distribution.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0])[1];
    const alt = second && second[0] > 0.05 ? `next: ${second[1] < entities.length ? `fill ← ${entities[second[1]].label}` : ["check", "click", "skip"][second[1] - entities.length]} (${second[0].toFixed(2)})` : "";
    const cells = [`${d.element.role} “${d.element.label}”`, describe(d, entities), d.probability.toFixed(3)];
    cells.forEach((c, i) => { const td = document.createElement("td"); td.textContent = c; if (i === 2) td.className = "p"; tr.append(td); });
    if (alt) tr.children[1].append(Object.assign(document.createElement("span"), { className: "alt", textContent: alt }));
    if (!acting.has(d) && d.action !== "skip") tr.title = d.action === "click" ? "clicks run only on a Submit button, with allow submit" : "below the confidence threshold";
    tbody.append(tr);
  }
  $("plan-meta").textContent = `${plan.decisions.length} elements scored in ${plan.latencyMs.toFixed(1)} ms · ${plan.actions.length} actions planned (struck-through ones will not run)`;
}

$("plan").onclick = async () => {
  if (!model) return;
  const entities = extractEntities(doc.value);
  const { elements, nodes } = observe();
  const title = form.getAttribute("aria-label") ?? document.title;
  const plan = await model.plan(title, elements, entities, { minConfidence: Number($<HTMLInputElement>("min").value), allowSubmit: $<HTMLInputElement>("submit").checked });
  last = { plan, entities, nodes };
  render(plan, entities);
  $<HTMLButtonElement>("apply").disabled = plan.actions.length === 0;
};

$("apply").onclick = () => {
  if (!last) return;
  for (const d of last.plan.actions) {
    const n = last.nodes.get(d.element.token!);
    if (!n) continue;
    if (d.action === "fill" && (n instanceof HTMLInputElement || n instanceof HTMLTextAreaElement)) {
      n.value = last.entities[d.entityIndex!].value;
      n.dispatchEvent(new Event("input", { bubbles: true })); n.dispatchEvent(new Event("change", { bubbles: true }));
      n.classList.add("filled");
    } else if (d.action === "check" && n instanceof HTMLInputElement && n.type === "checkbox" && !n.checked) {
      n.click();   // a real click, so the page's own handlers run
    } else if (d.action === "click" && n instanceof HTMLButtonElement) {
      form.requestSubmit(n.type === "submit" ? n : undefined);
    }
  }
  $<HTMLButtonElement>("apply").disabled = true;
};

form.onsubmit = (e) => { e.preventDefault(); status("Submitted — this is a demo, nothing was sent anywhere."); };
$("reset").onclick = () => { form.reset(); form.querySelectorAll(".filled").forEach((n) => n.classList.remove("filled")); $("decisions").querySelector("tbody")!.innerHTML = ""; $("plan-meta").textContent = ""; last = null; };

$<HTMLInputElement>("pdf").onchange = async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).href;
  const pdf = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;   // pdf.js >= 6.2.108: GHSA-hq66-cqwq-w95j
  const lines: string[] = [];
  for (let i = 1; i <= Math.min(pdf.numPages, 20); i++) {
    const content = await (await pdf.getPage(i)).getTextContent();
    const rows = new Map<number, { x: number; s: string }[]>();   // text runs grouped into lines by baseline
    for (const item of content.items) {
      if (!("str" in item) || !item.str.trim()) continue;
      const y = Math.round(item.transform[5] / 2) * 2;
      (rows.get(y) ?? rows.set(y, []).get(y)!).push({ x: item.transform[4], s: item.str });
    }
    [...rows.entries()].sort((a, b) => b[0] - a[0]).forEach(([, runs]) => lines.push(runs.sort((a, b) => a.x - b.x).map((r) => r.s).join(" ").replace(/\s+/g, " ").trim()));
  }
  doc.value = lines.join("\n");
  showEntities();
};

const t0 = performance.now();
model = await loadCuaS1(new URL(`${MODEL_BASE.replace(/\/$/, "")}/cua-s1-forms`, location.href).href, { ort: ort as unknown as OrtModule });
status(`cua-s1-forms loaded in ${Math.round(performance.now() - t0)} ms · ${model.manifest.source.split("@")[0]}@${model.manifest.source.split("@")[1].slice(0, 7)} · WASM${model.sharedOptions ? ", options encoded once per plan" : ""}`);
$<HTMLButtonElement>("plan").disabled = false;
