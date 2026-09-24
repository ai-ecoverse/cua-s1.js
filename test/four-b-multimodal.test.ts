// cua-s1-4b's multimodal adapter against upstream (fixtures/cua-s1-4b-0.2-multimodal.json and its GUI-360 screenshots,
// from `export/four_b/fixtures.py --modality multimodal`): the chat text, Qwen's image preprocessing and the mRoPE
// positions always; with the bundle in public/models/cua-s1-4b-0.2-multimodal (not in git), the token ids and the
// probabilities through the vision graph and the decoder on onnxruntime-node.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-node";
import { parseSafetensors } from "../src/fetch.ts";
import { CuaS1FourB, renderChat, type FourBContext, type FourBManifest } from "../src/four-b.ts";
import { preprocess, ropePositions, smartResize, visionInputs, type VisionConfig } from "../src/four-b-vision.ts";
import type { OrtModule } from "../src/model.ts";
import { options, quality, type Fixture as TaskFixture } from "./four-b-fixtures.ts";
import { decodePng } from "./png.ts";

interface Fixture extends TaskFixture {
  screenshot: string; size: [number, number]; grid_thw: [number, number, number];
  pixels: { shape: [number, number]; sum: number; sum_sq: number; head: number[]; tail: number[] };
}
const dir = new URL("../fixtures/", import.meta.url);
const fx: { run: string; fixtures: Fixture[] } = JSON.parse(readFileSync(new URL("cua-s1-4b-0.2-multimodal.json", dir), "utf8"));
const screenshot = (f: Fixture) => decodePng(readFileSync(new URL(f.screenshot, dir)));
const context = (f: Fixture): FourBContext => ({ app: f.app, taskFamily: f.family, screenshot: screenshot(f), goal: f.goal ?? undefined });

/** Qwen3.5-4B's preprocessor_config.json and vision_config, as export/four_b/vision.py writes them into the bundle. */
const VISION: VisionConfig = {
  patch_size: 16, merge_size: 2, temporal_patch_size: 2, hidden_size: 1024, num_heads: 16, num_grid_per_side: 48,
  rope_theta: 10000, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5], min_pixels: 65536, max_pixels: 16777216,
};

test("renderChat reproduces the processor's chat text", () => {
  for (const f of fx.fixtures) assert.equal(renderChat(options(f), context(f), "multimodal"), f.chat, f.id);
});

test("preprocess reproduces Qwen's pixel_values", () => {
  let worstMean = 0;
  for (const f of fx.fixtures) {
    const img = screenshot(f);
    assert.deepEqual([img.width, img.height], f.size, f.id);
    const p = preprocess(img, VISION);
    assert.deepEqual([1, p.gridH, p.gridW], f.grid_thw, f.id);
    assert.deepEqual([p.data.length / f.pixels.shape[1], f.pixels.shape[1]], f.pixels.shape, f.id);
    let sum = 0, sq = 0;
    for (const x of p.data) { sum += x; sq += x * x; }
    // GUI-360's screenshots are 1040 px wide, so the processor resizes them (bicubic, to 1024): allow a resampler
    // that is not torchvision's to differ by a rounding step here and there
    worstMean = Math.max(worstMean, Math.abs(sum - f.pixels.sum) / p.data.length);
    assert.ok(Math.abs(sq - f.pixels.sum_sq) / p.data.length < 2e-3, `${f.id}: sum of squares ${sq} vs ${f.pixels.sum_sq}`);
    f.pixels.head.forEach((v, i) => assert.ok(Math.abs(p.data[i] - v) < 0.02, `${f.id}: head[${i}] ${p.data[i]} vs ${v}`));
  }
  console.log(`mean |pixel difference| per value at most ${worstMean.toExponential(2)} (one uint8 step is ${(2 / 255).toExponential(2)})`);
  assert.ok(worstMean < 2e-3);
});

test("smartResize and mRoPE positions", () => {
  assert.deepEqual(smartResize(632, 760, VISION), [640, 768]);
  assert.deepEqual(smartResize(736, 1040, VISION), [736, 1024]);   // GUI-360
  assert.deepEqual(smartResize(100, 100, VISION), [256, 256]);        // below min_pixels
  const [t, h, w] = ropePositions([1, 2, 9, 9, 9, 9, 9, 9, 3], 4, 6, 9);   // 2 x 3 merged grid after two text tokens
  assert.deepEqual(t, [0, 1, 2, 2, 2, 2, 2, 2, 5]);
  assert.deepEqual(h, [0, 1, 2, 2, 2, 3, 3, 3, 5]);
  assert.deepEqual(w, [0, 1, 2, 3, 4, 2, 3, 4, 5]);
  const vi = visionInputs(4, 4, VISION);
  // 4 x 4 patches: the first (block order) is patch (0, 0), the table's corner; the last is (3, 3), the opposite corner
  assert.deepEqual(Array.from(vi.posIdx.subarray(0, 4), Number), [0, 1, 48, 49]);
  assert.deepEqual(Array.from(vi.posW.subarray(0, 4)), [1, 0, 0, 0]);
  assert.deepEqual(Array.from(vi.posIdx.subarray(60, 64), Number), [2303, 2303, 2303, 2303]);
  assert.equal(vi.posW[60], 1);
  assert.ok(vi.cos.subarray(0, 64).every((c) => c === 1) && vi.sin.subarray(0, 64).every((x) => x === 0));
});

const bundle = new URL("../public/models/cua-s1-4b-0.2-multimodal/", import.meta.url);
const manifest: FourBManifest | null = existsSync(new URL("manifest.json", bundle)) ? JSON.parse(readFileSync(new URL("manifest.json", bundle), "utf8")) : null;
const skip = manifest ? false : "no model bundle in public/models/cua-s1-4b-0.2-multimodal (export/build_4b_model.sh --multimodal)";

test("probabilities match FourBModel through the vision graph", { skip }, async () => {
  assert.equal(fx.run, manifest!.run, "fixtures and weights come from the same adapter commit");
  const { image_mean, image_std, min_pixels, max_pixels, rope_theta, num_grid_per_side } = manifest!.vision!.config;
  assert.deepEqual({ image_mean, image_std, min_pixels, max_pixels, rope_theta, num_grid_per_side },
    { image_mean: VISION.image_mean, image_std: VISION.image_std, min_pixels: VISION.min_pixels, max_pixels: VISION.max_pixels, rope_theta: VISION.rope_theta, num_grid_per_side: VISION.num_grid_per_side });
  const variant = Object.keys(manifest!.variants)[0], v = manifest!.variants[variant];
  const rd = (p: string) => readFileSync(new URL(p, bundle), "utf8");
  const m = new CuaS1FourB({
    ort: ort as unknown as OrtModule, manifest: manifest!, variant,
    session: await ort.InferenceSession.create(new URL(v.model, bundle).pathname),
    vision: await ort.InferenceSession.create(new URL(manifest!.vision!.model, bundle).pathname),
    head: parseSafetensors(readFileSync(new URL(manifest!.files.head, bundle)).buffer as ArrayBuffer).weight.data,
    tokenizer: new Tokenizer(JSON.parse(rd(manifest!.files.tokenizer)), JSON.parse(rd(manifest!.files.tokenizer_config))),
  });
  let worst = 0, flips = 0;
  const got: number[][] = [];
  for (const f of fx.fixtures) {
    const img = await m.imageEmbeds(screenshot(f));
    assert.deepEqual(m.encode(options(f), context(f), img.embeds.length / manifest!.hidden_size), f.input_ids, f.id);
    const r = await m.score(options(f), context(f));
    const p = r.options.map((o) => o.probability);
    f.probs.forEach((q, k) => { worst = Math.max(worst, Math.abs(q - p[k])); });
    if (p.indexOf(Math.max(...p)) !== f.probs.indexOf(Math.max(...f.probs))) flips++;
    got.push(p);
  }
  console.log(`${variant}: ${fx.fixtures.length} tasks, max |dp| ${worst.toExponential(2)}, ${flips} argmax flips; ${quality(fx.fixtures, got)}`
    + ` (FourBModel: ${quality(fx.fixtures, fx.fixtures.map((f) => f.probs))})`);
  // the bundle's own parity uses Qwen's processor; this goes through preprocess(), whose resize differs from
  // torchvision's by a rounding step on a few pixels
  assert.ok(worst <= (v.parity?.max_abs_dp ?? 0.05) + 0.03, `max |dp| ${worst}`);
  await m.release();
});
