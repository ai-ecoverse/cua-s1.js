// The ONNX graph through the JS runtime (collation included) against PyTorch probabilities (fixtures from export.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-node";
import { CuaS1, type Manifest, type OrtModule } from "../src/index.ts";

const dir = new URL("../public/models/cua-s1-forms/", import.meta.url);
const manifest: Manifest = JSON.parse(readFileSync(new URL("manifest.json", dir), "utf8"));
const fx = JSON.parse(readFileSync(new URL("../fixtures/cua-s1-forms.json", import.meta.url), "utf8"));

for (const sharedOptions of [false, true]) {
  test(`probabilities match PyTorch on every synthetic decision (${sharedOptions ? "shared-options" : "per-row"} graph)`, async () => {
    assert.equal(fx.source, manifest.source, "fixtures and model come from the same checkpoint");
    const file = sharedOptions ? manifest.shared_options! : manifest;
    const session = await ort.InferenceSession.create(new URL(file.model, dir).pathname);
    const kev = new CuaS1({ ort: ort as unknown as OrtModule, session, manifest, sharedOptions });
    let worst = 0, flips = 0;
    for (let i = 0; i < fx.fixtures.length; i += 16) {   // mixed lengths and option counts per batch
      const chunk = fx.fixtures.slice(i, i + 16);
      const got = await kev.score(chunk);
      chunk.forEach((f: { probs: number[] }, j: number) => {
        assert.equal(got[j].length, f.probs.length);
        f.probs.forEach((p, k) => { worst = Math.max(worst, Math.abs(p - got[j][k])); });
        const am = (a: number[]) => a.indexOf(Math.max(...a));
        if (am(f.probs) !== am(got[j])) flips++;
      });
    }
    console.log(`${fx.fixtures.length} decisions: max |dp| ${worst.toExponential(2)}, ${flips} argmax flips`);
    assert.ok(worst < 1e-4, `max |dp| ${worst}`);
    assert.equal(flips, 0);
    await kev.release();
  });
}
