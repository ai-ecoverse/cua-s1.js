// Everything around the model, compared exactly against cua_s1's Python (fixtures/schema.json, schema_fixtures.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { collate, extractEntities, filterElements, isSubmitControl, normalizeTitle, orderDecisions, renderContext, renderOptions, type Action, type Decision } from "../src/index.ts";

const fx = JSON.parse(readFileSync(new URL("../fixtures/schema.json", import.meta.url), "utf8"));
const el = (e: Record<string, unknown>) => ({ role: e.role as string, label: e.label as string, value: e.value as string, checked: e.checked as boolean | null });

test("renderContext matches render_context, including code-point truncation", () => {
  for (const c of fx.contexts) assert.equal(renderContext(c.title, el(c.element), c.placeholder), c.context);
});

test("renderOptions matches render_options", () => assert.deepEqual(renderOptions(fx.entities), fx.options));

test("collate matches ByteCollator tensor for tensor", () => {
  const b = collate(fx.collation.examples, 224, 96);
  const t = fx.collation.tensors;
  const flat = (x: unknown): number[] => (Array.isArray(x) ? x.flatMap(flat) : [Number(x)]);
  assert.deepEqual([b.batch, b.contextLen], [t.context_ids.length, t.context_ids[0].length]);
  assert.deepEqual([b.options, b.optionLen], [t.option_ids[0].length, t.option_ids[0][0].length]);
  assert.deepEqual(Array.from(b.contextIds, Number), flat(t.context_ids));
  assert.deepEqual(Array.from(b.optionIds, Number), flat(t.option_ids));
  assert.deepEqual(Array.from(b.contextMask), flat(t.context_mask));
  assert.deepEqual(Array.from(b.optionTokenMask), flat(t.option_token_mask));
  assert.deepEqual(Array.from(b.optionMask), flat(t.option_mask));
});

test("extractEntities matches pdf.extract_entities' line parsing", () => assert.deepEqual(extractEntities(fx.document), fx.extracted));

test("orderDecisions matches order_decisions", () => {
  const decisions: Decision[] = fx.decisions.map((d: { index: number; role: string; label: string; action: Action; probability: number }) =>
    ({ element: { role: d.role, label: d.label, index: d.index }, action: d.action, entityIndex: d.action === "fill" ? 0 : null, probability: d.probability, distribution: [] }));
  for (const o of fx.ordering)
    assert.deepEqual(orderDecisions(decisions, o.min_confidence, { allowSubmit: o.allow_submit }).map((d) => d.element.index), o.order, JSON.stringify(o));
});

test("titles, actionable roles and submit controls", () => {
  for (const t of fx.titles) assert.equal(normalizeTitle(t.in), t.out);
  const els = fx.contexts.slice(0, 13 * 3).filter((_: unknown, i: number) => i % 3 === 0).map((c: { element: Record<string, unknown> }) => el(c.element));
  assert.deepEqual(filterElements(els).map((e) => e.role), fx.actionable);
  for (const s of fx.submit) assert.equal(isSubmitControl({ role: s.role, label: s.label }), s.is_submit, `${s.role} ${s.label}`);
});
