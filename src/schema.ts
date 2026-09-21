// Inputs, option encoding and plan ordering for cua-s1 form decisions: a port of cua_s1/schema.py and the
// model-independent parts of cua_s1/planner.py and cua_s1/pdf.py (trycua/cua, libs/cua-s1). The model was trained on
// exactly these strings, so rendering follows Python's semantics, including slicing by code point, not UTF-16 unit.

export type Action = "fill" | "check" | "click" | "skip";
export const FIXED_ACTIONS = ["check", "click", "skip"] as const;
/** Execution order for a single-pass plan: fills, checkboxes, then clicks. */
export const ACTION_ORDER: Record<Action, number> = { fill: 0, check: 1, click: 2, skip: 3 };

/** A labelled source-document value the model may point to. */
export interface Entity { label: string; value: string }

/** A normalized UI element observation. `index` orders the plan; `token` identifies the element to act on. */
export interface Element {
  role: string;
  label: string;
  value?: string;
  checked?: boolean | null;
  index?: number;
  token?: string;
  /** placeholder text, rendered as hint="..." */
  placeholder?: string;
}

export interface Decision {
  element: Element;
  action: Action;
  /** index into the entities for a fill */
  entityIndex: number | null;
  /** probability of the chosen option */
  probability: number;
  /** probability of every option: one per entity, then check, click, skip */
  distribution: number[];
}

/** Python str[:n]: by code point. */
const head = (s: string, n: number) => { const cps = Array.from(s); return cps.length <= n ? s : cps.slice(0, n).join(""); };

export const entityOption = (e: Entity) => `fill ${e.label}: ${e.value}`;

/** The byte-level context the model scores one element in. */
export function renderContext(formTitle: string, element: Element, placeholder = element.placeholder ?? ""): string {
  const state = element.role === "CheckBox" ? (element.checked ? "checked" : "unchecked") : `value="${head(element.value ?? "", 48)}"`;
  const hint = placeholder ? ` hint="${head(placeholder, 72)}"` : "";
  return "TASK fill the form from the document, then submit\n"
    + `FORM ${head(formTitle, 64)}\n`
    + `ELEMENT ${element.role} "${head(element.label, 72)}" ${state}${hint}`;
}

/** Entity pointer options followed by the fixed actions. */
export const renderOptions = (entities: Entity[]): string[] => [...entities.map(entityOption), ...FIXED_ACTIONS];

export function decode(optionIndex: number, entities: Entity[]): { action: Action; entityIndex: number | null } {
  const count = entities.length + FIXED_ACTIONS.length;
  if (!Number.isInteger(optionIndex) || optionIndex < 0 || optionIndex >= count)
    throw new RangeError(`option index ${optionIndex} is outside the valid range 0..${count - 1}`);
  return optionIndex < entities.length
    ? { action: "fill", entityIndex: optionIndex }
    : { action: FIXED_ACTIONS[optionIndex - entities.length], entityIndex: null };
}

export function encode(action: Action, entityIndex: number | null, entities: Entity[]): number {
  if (action === "fill") {
    if (entityIndex === null || !Number.isInteger(entityIndex) || entityIndex < 0 || entityIndex >= entities.length)
      throw new RangeError("fill actions require a valid entity index");
    return entityIndex;
  }
  if (entityIndex !== null) throw new RangeError(`${action} actions must not include an entity index`);
  const i = (FIXED_ACTIONS as readonly string[]).indexOf(action);
  if (i < 0) throw new RangeError(`unsupported action: ${action}`);
  return entities.length + i;
}

const ACTIONABLE_ROLES = new Set(["button", "checkbox", "combobox", "edit", "textfield", "axbutton", "axcheckbox", "axcombobox", "axtextfield"]);
const SUBMIT_ROLES = new Set(["button", "axbutton"]);
const SUBMIT_LABELS = new Set(["submit", "submit form"]);
const APP_SUFFIXES = [" - Google Chrome", " - Microsoft Edge", " - Mozilla Firefox", " - Brave", " - Safari"];

const normalizedRole = (role: string) => role.replaceAll("_", "").replaceAll(" ", "").toLowerCase();
const normalizedLabel = (label: string) => (label.toLowerCase().match(/[a-z0-9]+/g) ?? []).join(" ");

/** Window titles carry the browser name; the model was trained on the page title alone. */
export function normalizeTitle(title: string): string {
  for (const s of APP_SUFFIXES) if (title.endsWith(s)) return title.slice(0, -s.length);
  return title;
}

export const filterElements = (elements: Element[]) => elements.filter((e) => ACTIONABLE_ROLES.has(normalizedRole(e.role)));

/** Only a button labelled exactly "Submit" or "Submit Form" may be clicked. */
export const isSubmitControl = (e: Element) => SUBMIT_ROLES.has(normalizedRole(e.role)) && SUBMIT_LABELS.has(normalizedLabel(e.label));

/**
 * Keep confident, non-skip decisions and order them fills, checkboxes, clicks. At most one click survives, and only
 * a recognized submit control with allowSubmit (cua_s1.planner.order_decisions).
 */
export function orderDecisions(decisions: Decision[], minConfidence: number, o: { allowSubmit?: boolean } = {}): Decision[] {
  if (!Number.isFinite(minConfidence) || minConfidence < 0 || minConfidence > 1) throw new RangeError("minConfidence must be between zero and one");
  let keep = decisions.filter((d) => d.action !== "skip" && d.probability >= minConfidence);
  const submits = keep.filter((d) => d.action === "click" && isSubmitControl(d.element)).sort((a, b) => b.probability - a.probability);
  keep = keep.filter((d) => d.action !== "click");
  if (o.allowSubmit && submits.length) keep.push(submits[0]);
  return keep.sort((a, b) => ACTION_ORDER[a.action] - ACTION_ORDER[b.action] || (a.element.index ?? -1) - (b.element.index ?? -1));
}

const LINE = /^\s*([A-Za-z][A-Za-z0-9 .'/#&()-]{1,40}?)\s*[:–-]\s+(.+?)\s*$/;

/** `Label: value` pairs from document text, one per line, deduplicated (cua_s1.pdf.extract_entities). */
export function extractEntities(text: string, maxValueLength = 120): Entity[] {
  const out: Entity[] = [];
  const seen = new Set<string>();
  for (const line of text.split(/\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/)) {   // Python str.splitlines()
    const m = LINE.exec(line);
    if (!m) continue;
    const label = m[1].trim(), value = m[2].trim();
    const key = JSON.stringify([label, value]);
    if (Array.from(value).length > maxValueLength || seen.has(key)) continue;
    seen.add(key);
    out.push({ label, value });
  }
  return out;
}
