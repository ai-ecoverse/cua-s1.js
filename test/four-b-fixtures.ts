// What export/four_b/fixtures.py writes, and the two ways it counts a screen as right (its task_correct and
// action_chosen).
import { elementDecisions, LETTERS, type FourBOption } from "../src/four-b.ts";

export interface Fixture {
  id: string; family: string; app: string; goal: string | null; ax_tree: string | null; gold: string[];
  chat: string; input_ids: number[]; probs: number[];
  options: { element_id: string; role: string; label: string; action: string; entity_id: string | null }[];
}

export const options = (f: Fixture): FourBOption[] =>
  f.options.map((o) => ({ elementId: o.element_id, role: o.role, label: o.label, action: o.action, entityId: o.entity_id }));

/** cua-bench-s1's task accuracy: every element's likeliest action is its gold one (skip unless the key says otherwise). */
export function taskCorrect(f: Fixture, probs: number[]): boolean {
  const opts = options(f);
  const gold = new Map(f.gold.map((l) => { const o = opts[LETTERS.indexOf(l)]; return [o.elementId, o.action]; }));
  const chosen = elementDecisions(opts.map((option, i) => ({ letter: LETTERS[i], option, probability: probs[i] })));
  return [...chosen].every(([id, o]) => o.option.action === (gold.get(id) ?? "skip"));
}

/** The recorded action wins on its own element: the model would take it (whatever it decides elsewhere). */
export function actionChosen(f: Fixture, probs: number[]): boolean {
  const opts = options(f);
  const chosen = elementDecisions(opts.map((option, i) => ({ letter: LETTERS[i], option, probability: probs[i] })));
  return f.gold.every((l) => { const o = opts[LETTERS.indexOf(l)]; return o.action === "skip" || chosen.get(o.elementId)!.letter === l; });
}

export function quality(fixtures: Fixture[], probs: number[][]): string {
  const n = fixtures.length, count = (fn: (f: Fixture, p: number[]) => boolean) => fixtures.filter((f, i) => fn(f, probs[i])).length;
  return `task accuracy ${count(taskCorrect)}/${n}, recorded action chosen ${count(actionChosen)}/${n}`;
}
