// loadCuaS1 over HTTP with onnxruntime-web's WASM backend, as in a browser: it picks the shared-options graph, which
// must plan exactly what the per-row graph plans.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import * as ort from "onnxruntime-web";
import { extractEntities, loadCuaS1, type Element, type OrtModule } from "../src/index.ts";

ort.env.wasm.numThreads = 1;
const models = new URL("../public/models/", import.meta.url);

const elements: Element[] = [
  ...["First name", "Last name", "Date of birth", "Phone number", "Email address", "Street address", "City", "ZIP code",
    "Emergency contact phone", "Insurance member ID", "Known allergies"].map((label, index) => ({ role: "Edit", label, value: "", index, token: `el-${index}` })),
  { role: "CheckBox", label: "I consent to treatment", checked: false, index: 11, token: "el-11" },
  { role: "CheckBox", label: "Send me the newsletter", checked: false, index: 12, token: "el-12" },
  { role: "Button", label: "Save draft", index: 13, token: "el-13" },
  { role: "Button", label: "Submit", index: 14, token: "el-14" },
];
const entities = extractEntities(`First name: Amara\nLast name: Ivanova\nDOB: 03/14/1987\nTel: (503) 555-0142
Email: amara.ivanova@example.invalid\nStreet: 4881 Station Court\nCity: Portland\nZIP: 97205
Emergency contact phone: (503) 555-0199\nMember ID: NWC-448-2291\nAllergies: Latex, penicillin\nWork phone: (503) 555-0110`);

test("loadCuaS1 plans the same with the shared-options and the per-row graph", async () => {
  const server = createServer(async (req, res) => {
    try { res.end(await readFile(new URL(req.url!.slice(1), models))); } catch { res.statusCode = 404; res.end(); }
  }).listen(0, "127.0.0.1");
  await new Promise((r) => server.once("listening", r));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/cua-s1-forms`;
  try {
    const shared = await loadCuaS1(base, { ort: ort as unknown as OrtModule });
    const rows = await loadCuaS1(base, { ort: ort as unknown as OrtModule, sharedOptions: false });
    assert.equal(shared.sharedOptions, true);
    assert.equal(rows.sharedOptions, false);
    const a = await shared.plan("Northwind Clinic - New Patient Registration", elements, entities, { allowSubmit: true });
    const b = await rows.plan("Northwind Clinic - New Patient Registration", elements, entities, { allowSubmit: true });
    assert.equal(a.decisions.length, elements.length);
    a.decisions.forEach((d, i) => {
      assert.equal(d.action, b.decisions[i].action);
      assert.equal(d.entityIndex, b.decisions[i].entityIndex);
      d.distribution.forEach((p, k) => assert.ok(Math.abs(p - b.decisions[i].distribution[k]) < 1e-5));
    });
    assert.deepEqual(a.actions.map((d) => d.element.token), b.actions.map((d) => d.element.token));
    const phone = a.decisions.find((d) => d.element.label === "Phone number")!;
    assert.equal(phone.action, "fill");
    assert.equal(entities[phone.entityIndex!].label, "Tel");
    await assert.rejects(shared.score([{ context: "x", options: [] }]), /empty batch/);
    await shared.release(); await rows.release();
  } finally {
    server.close();
  }
});
