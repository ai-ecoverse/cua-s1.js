"""Reference outputs from cua_s1's own Python for the JS port of everything around the model: context and option
rendering, byte collation, entity extraction, title normalization and plan ordering.

    uv run python schema_fixtures.py > ../fixtures/schema.json
"""
import json, re, sys
from cua_s1.model import ByteCollator, ChoiceExample
from cua_s1.pdf import LINE, DEFAULT_MAX_VALUE_LENGTH
from cua_s1.planner import filter_elements, normalize_title, order_decisions, _is_submit_control
from cua_s1.schema import Decision, Element, Entity, render_context, render_options

LONG = "Very long label " * 8
EMOJI = "Name 😀" * 20      # astral characters straddle every truncation limit
elements = [
    Element("Edit", "Phone number"),
    Element("Edit", "Email", value="old@example.invalid"),
    Element("CheckBox", "I agree to the terms", checked=False),
    Element("CheckBox", "Subscribe", checked=True),
    Element("Button", "Submit"),
    Element("Button", "Submit form!"),
    Element("AXButton", "  SUBMIT  "),
    Element("Button", "Save draft"),
    Element("ComboBox", "Country", value="Germany"),
    Element("Edit", LONG, value=LONG),
    Element("Edit", EMOJI, value=EMOJI),
    Element("text_field", "Middle name"),
    Element("Static Text", "Instructions"),
]
contexts = [{"title": t, "element": e.__dict__, "placeholder": p, "context": render_context(t, e, p)}
            for t in ["Northwind Clinic - New Patient Registration", "Ünïcödé 表单 " * 10]
            for e in elements for p in ["", "(555) 555-5555", "x" * 100]]

entities = [Entity("Tel", "(503) 555-0142"), Entity("DOB", "03/14/1987"), Entity("Name", EMOJI)]
options = render_options(entities)

collator = ByteCollator(224, 96)
texts = [("TASK aé\U0001F600", ["fill x: y", "check", "click", "skip"]),
         ("lone \ud800 surrogate", ["fill \udc00 z: 1", "skip"]),
         (EMOJI * 3, [EMOJI * 2, "check"]),
         ("", ["", "click"])]
batch = collator([ChoiceExample(c, tuple(o), 0) for c, o in texts])
collation = {"examples": [{"context": c, "options": o} for c, o in texts],
             "tensors": {k: v.tolist() for k, v in batch.items() if k != "labels"}}

doc = ("Patient record\nName: Amara Ivanova\nDOB – 03/14/1987\n  Tel:   (503) 555-0142  \nTel: (503) 555-0142\n"
       "no colon here\nNotes: " + "x" * 130 + "\nEmergency contact phone - +1 202 555 0161 Blood group: A-\x0bAllergies: Latex\x85"
       "1st line: starts with a digit\nA: too short a label")
from cua_s1.pdf import Entity as _E  # noqa: F401  (same class)
extracted = []
seen = set()
for line in doc.splitlines():
    m = LINE.match(line)
    if not m: continue
    label, value = m.group(1).strip(), m.group(2).strip()
    if len(value) > DEFAULT_MAX_VALUE_LENGTH or (label, value) in seen: continue
    seen.add((label, value)); extracted.append({"label": label, "value": value})

els = [Element(r, l, index=i) for i, (r, l) in enumerate([("Edit", "A"), ("CheckBox", "B"), ("Button", "Submit"), ("Button", "Submit Form"), ("Edit", "C"), ("Button", "Cancel")])]
acts = ["fill", "check", "click", "click", "skip", "click"]
probs = [0.9, 0.7, 0.8, 0.95, 0.99, 0.6]
decisions = [Decision(e, a, 0 if a == "fill" else None, p) for e, a, p in zip(els, acts, probs)]
ordering = [{"min_confidence": mc, "allow_submit": s,
             "order": [d.element.index for d in order_decisions(decisions, mc, allow_submit=s)]}
            for mc in [0.0, 0.75, 0.96] for s in [False, True]]

json.dump({
    "contexts": contexts, "entities": [e.__dict__ for e in entities], "options": options, "collation": collation,
    "document": doc, "extracted": extracted,
    "decisions": [{"index": d.element.index, "role": d.element.role, "label": d.element.label, "action": d.action, "probability": d.probability} for d in decisions],
    "ordering": ordering,
    "titles": [{"in": t, "out": normalize_title(t)} for t in ["Form - Google Chrome", "Form - Safari - Safari", "Form"]],
    "actionable": [e.role for e in filter_elements(elements)],
    "submit": [{"role": e.role, "label": e.label, "is_submit": _is_submit_control(e)} for e in elements],
}, sys.stdout, ensure_ascii=True)   # escapes lone surrogates; JSON.parse restores them
