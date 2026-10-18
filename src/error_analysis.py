"""Phase 5: qualitative before-vs-after examples and error categorisation of the fine-tuned model.

Entity-level categories (fine-tuned predictions vs gold, test split):
  correct     exact span and type
  type        exact span, wrong type
  boundary    overlapping span, same type, different boundaries
  bound+type  overlapping span, different type and boundaries
  missed      gold entity with no overlapping prediction
  spurious    predicted entity with no overlapping gold entity
Outputs results/error_analysis.json, results/qualitative_examples.json, results/tables/*.md
"""
from collections import Counter, defaultdict

from common import ENTITY_TYPES, RESULTS, bio_to_spans, load_json, load_split, read_jsonl, save_json


def categorise(gold, pred):
    cats, per_type, examples = Counter(), defaultdict(Counter), defaultdict(list)
    matched_pred = set()
    for g in gold:
        gs, ge, gt = g
        exact = [p for p in pred if p[0] == gs and p[1] == ge]
        over = [p for p in pred if p[0] < ge and p[1] > gs]
        if exact and exact[0][2] == gt:
            c, p = "correct", exact[0]
        elif exact:
            c, p = "type", exact[0]
        elif any(p[2] == gt for p in over):
            c, p = "boundary", next(p for p in over if p[2] == gt)
        elif over:
            c, p = "bound+type", over[0]
        else:
            c, p = "missed", None
        matched_pred.update(over)
        cats[c] += 1
        per_type[gt][c] += 1
        examples[c].append((g, p))
    for p in pred:
        if p not in matched_pred and not any(g[0] < p[1] and g[1] > p[0] for g in gold):
            cats["spurious"] += 1
            per_type[p[2]]["spurious"] += 1
            examples["spurious"].append((None, p))
    return cats, per_type, examples


def snippet_entities(text, spans, a, b):
    return [{"text": text[s:e], "label": t} for s, e, t in spans if s >= a and e <= b]


def main():
    splits, _ = load_split("clean")
    test = sorted(splits["test"], key=lambda d: d["id"])
    final_tag = load_json(RESULTS / "final_config.json")["final_run"]
    pre = {r["id"]: r["tags"] for r in read_jsonl(RESULTS / "predictions" / "pretrained_test.jsonl")}
    ft = {r["id"]: r for r in read_jsonl(RESULTS / "predictions" / f"{final_tag}_test.jsonl")}

    tot, per_type, ex_all = Counter(), defaultdict(Counter), defaultdict(list)
    for d in test:
        gold = [tuple(s) for s in d["spans"]]
        pred = bio_to_spans(ft[d["id"]]["tags"], d["starts"], d["ends"])
        c, pt, ex = categorise(gold, pred)
        tot += c
        for t, v in pt.items():
            per_type[t] += v
        for k, v in ex.items():
            for g, p in v[:3]:
                if len(ex_all[k]) < 6:
                    ex_all[k].append({"doc": d["id"],
                                      "gold": None if g is None else {"text": d["text"][g[0]:g[1]], "label": g[2]},
                                      "pred": None if p is None else {"text": d["text"][p[0]:p[1]], "label": p[2]}})
    final = load_json(RESULTS / "runs" / f"{final_tag}.json")
    f1 = {t: final["test"]["per_type"][t]["f1"] for t in ENTITY_TYPES}
    hardest = sorted(ENTITY_TYPES, key=lambda t: f1[t])
    save_json(RESULTS / "error_analysis.json", {
        "model": final_tag, "categories": dict(tot), "per_type": {t: dict(per_type[t]) for t in ENTITY_TYPES},
        "hardest_types_by_f1": [{"type": t, "f1": f1[t]} for t in hardest], "examples": ex_all})

    cols = ["correct", "boundary", "type", "bound+type", "missed", "spurious"]
    md = "| Entity type | F1 | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2) + "\n"
    for t in hardest:
        md += f"| {t} | {100 * f1[t]:.1f} | " + " | ".join(str(per_type[t][c]) for c in cols) + " |\n"
    md += "| **Total** | | " + " | ".join(str(tot[c]) for c in cols) + " |\n"
    (RESULTS / "tables").mkdir(exist_ok=True)
    (RESULTS / "tables" / "error_analysis.md").write_text(md, encoding="utf-8")

    # qualitative: 10 snippets (header / experience / education), same text, gold vs both models
    qual = []
    kinds = ["header", "DESIGNATION", "DEGREE"]
    for i, d in enumerate(test[:10]):
        kind = kinds[i % 3]
        text = d["text"]
        if kind == "header":
            a = 0
        else:
            first = next((s for s in d["spans"] if s[2] == kind), None)
            if first is None:
                kind, a = "header", 0
            else:
                a = text.rfind("\n", 0, max(0, first[0] - 40)) + 1
        b = min(len(text), a + 260)
        nl = text.rfind("\n", a, b)
        b = nl if nl > a + 120 else b
        gold = [tuple(s) for s in d["spans"]]
        pspans = bio_to_spans(pre[d["id"]], d["starts"], d["ends"])
        fspans = bio_to_spans(ft[d["id"]]["tags"], d["starts"], d["ends"])
        qual.append({"doc": d["id"], "section": kind, "snippet": text[a:b], "start": a, "end": b,
                     "gold": snippet_entities(text, gold, a, b),
                     "pretrained": snippet_entities(text, pspans, a, b),
                     "finetuned": snippet_entities(text, fspans, a, b)})
    save_json(RESULTS / "qualitative_examples.json", qual)
    fmt = lambda es: "; ".join(f"{e['text']!r}→{e['label']}" for e in es) or "—"
    mdq = ""
    for q in qual:
        mdq += (f"### {q['doc']} ({q['section']})\n```\n{q['snippet'].strip()}\n```\n"
                f"- **Gold:** {fmt(q['gold'])}\n- **Pretrained:** {fmt(q['pretrained'])}\n"
                f"- **Fine-tuned:** {fmt(q['finetuned'])}\n\n")
    (RESULTS / "tables" / "qualitative.md").write_text(mdq, encoding="utf-8")
    print(md)
    print("hardest:", [(t, round(f1[t], 3)) for t in hardest[:4]])


if __name__ == "__main__":
    main()
