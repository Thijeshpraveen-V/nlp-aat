"""Phase 1: load DataTurks resume JSON, clean text (offset-preserving), convert spans to
word-level BIO, de-duplicate, create 70/15/15 split + 5-fold CV, and write statistics.

Outputs
  data/processed/clean/docs.jsonl   cleaned text (main pipeline)
  data/processed/raw/docs.jsonl     un-normalised text (text-normalisation ablation)
  data/processed/splits.json        doc ids for train/val/test and 5 CV folds
  results/data_stats.json, results/figures/label_distribution.png
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split

from common import (DATA_PROCESSED, DATA_RAW, ENTITY_TYPES, FIGURES, RAW2TYPE, RESULTS, SEED,
                    bio_to_spans, save_json, set_seed, word_tokenize, write_jsonl)


def clean_text(text: str) -> str:
    """Length-preserving normalisation so annotation offsets stay valid:
    control chars, U+FFFD replacement chars, private-use bullets and odd spaces -> ' '."""
    out = []
    for ch in text:
        if ch in "\n":
            out.append(ch)
        elif ch == "�" or unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cs", "Zs", "Zl", "Zp"):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def load_raw():
    rows = []
    with open(DATA_RAW, encoding="utf-8") as f:
        for i, line in enumerate(l for l in f if l.strip()):
            rows.append(json.loads(line) | {"orig_index": i})
    return rows


def extract_spans(rec, text, log):
    """Return trimmed spans [start, end) with mapped type. DataTurks 'end' is inclusive."""
    spans = []
    for ann in rec["annotation"] or []:
        for lab in ann["label"]:
            if lab not in RAW2TYPE:
                log["dropped_unknown_label"] += 1
                continue
            for p in ann["points"]:
                s, e = p["start"], p["end"] + 1
                if text[s:e].strip() == "":
                    log["dropped_empty"] += 1
                    continue
                # trim whitespace / bullet noise at span edges
                while s < e and not text[s].strip():
                    s += 1
                while e > s and not text[e - 1].strip():
                    e -= 1
                spans.append((s, e, RAW2TYPE[lab]))
    return sorted(set(spans), key=lambda x: (x[0], -(x[1] - x[0])))


def resolve_overlaps(spans, log):
    """Greedy: sort by start (longer first), keep spans that do not overlap an already-kept span."""
    kept = []
    for s, e, t in spans:
        if kept and s < kept[-1][1]:
            log["dropped_overlap"] += 1
            log["dropped_overlap_by_type"][t] += 1
            continue
        kept.append((s, e, t))
    return kept


def spans_to_bio(spans, starts, ends, log):
    """Tag every word that overlaps a span. Spans whose edges fall inside a word are
    expanded to the full word boundary (logged as misaligned)."""
    tags = ["O"] * len(starts)
    aligned_spans = []
    for s, e, t in spans:
        idx = [i for i, (ws, we) in enumerate(zip(starts, ends)) if ws < e and we > s]
        if not idx:
            log["span_no_tokens"] += 1
            continue
        if starts[idx[0]] != s or ends[idx[-1]] != e:
            log["span_misaligned_expanded"] += 1
        for k, i in enumerate(idx):
            tags[i] = ("B-" if k == 0 else "I-") + t
        aligned_spans.append((starts[idx[0]], ends[idx[-1]], t))
    return tags, aligned_spans


def build_docs(raw_rows, variant, log):
    docs = []
    for r in raw_rows:
        text = r["content"]
        if variant == "clean":
            text = clean_text(text)
        assert len(text) == len(r["content"])
        spans = resolve_overlaps(extract_spans(r, text, log), log)
        toks, starts, ends = word_tokenize(text)
        tags, aligned = spans_to_bio(spans, starts, ends, log)
        # round-trip check
        recovered = set(bio_to_spans(tags, starts, ends))
        log["roundtrip_total"] += len(aligned)
        log["roundtrip_recovered"] += sum(1 for a in aligned if a in recovered)
        log["orig_exact_recovered"] += sum(1 for a in spans if a in recovered)
        log["orig_total"] += len(spans)
        docs.append({"id": r["id"], "text": text, "tokens": toks, "starts": starts, "ends": ends,
                     "tags": tags, "spans": [list(a) for a in aligned]})
    return docs


def new_log():
    return Counter(dropped_overlap_by_type=Counter()) | {"dropped_overlap_by_type": Counter()}


def main():
    set_seed()
    raw = load_raw()
    n_raw = len(raw)

    # de-duplicate on normalised content
    seen, dedup, dup_ids = {}, [], []
    for r in raw:
        key = hashlib.md5(re.sub(r"\s+", " ", r["content"]).strip().lower().encode()).hexdigest()
        if key in seen:
            dup_ids.append((r["orig_index"], seen[key]))
            continue
        seen[key] = r["orig_index"]
        dedup.append(r)
    for i, r in enumerate(dedup):
        r["id"] = f"resume_{r['orig_index']:03d}"

    logs = {}
    out = {}
    for variant in ("clean", "raw"):
        log = Counter()
        log["dropped_overlap_by_type"] = Counter()
        out[variant] = build_docs(dedup, variant, log)
        write_jsonl(DATA_PROCESSED / variant / "docs.jsonl", out[variant])
        logs[variant] = log

    # splits: 70/15/15 + 5-fold CV (inside each fold, 15% of non-test docs -> val)
    ids = [d["id"] for d in out["clean"]]
    train_ids, rest = train_test_split(ids, test_size=0.30, random_state=SEED)
    val_ids, test_ids = train_test_split(rest, test_size=0.50, random_state=SEED)
    folds = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=SEED).split(ids):
        tr_ids = [ids[i] for i in tr]
        f_train, f_val = train_test_split(tr_ids, test_size=0.15, random_state=SEED)
        folds.append({"train": f_train, "val": f_val, "test": [ids[i] for i in te]})
    # leakage checks
    assert not (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids))
    for f in folds:
        assert not (set(f["train"]) & set(f["test"]) or set(f["val"]) & set(f["test"]))
    text_by_id = {d["id"]: d["text"] for d in out["clean"]}
    text_overlap = len({text_by_id[i] for i in train_ids} & {text_by_id[i] for i in test_ids})
    save_json(DATA_PROCESSED / "splits.json",
              {"train": train_ids, "val": val_ids, "test": test_ids, "folds": folds})

    # statistics
    docs = out["clean"]
    log = logs["clean"]
    ent_counts = Counter(t for d in docs for _, _, t in d["spans"])
    tag_counts = Counter(t for d in docs for t in d["tags"])
    n_tokens = sum(len(d["tokens"]) for d in docs)
    split_ent = {name: dict(Counter(t for d in docs if d["id"] in set(sids) for _, _, t in d["spans"]))
                 for name, sids in (("train", train_ids), ("val", val_ids), ("test", test_ids))}
    stats = {
        "source": "https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner",
        "file": DATA_RAW.name,
        "n_resumes_raw": n_raw,
        "n_duplicates_removed": len(dup_ids),
        "duplicate_pairs_(orig_index,kept_index)": dup_ids,
        "n_resumes": len(docs),
        "n_word_tokens": n_tokens,
        "tokens_per_resume": {"mean": n_tokens / len(docs),
                              "min": min(len(d["tokens"]) for d in docs),
                              "max": max(len(d["tokens"]) for d in docs)},
        "entities_per_type": dict(ent_counts.most_common()),
        "entities_total": sum(ent_counts.values()),
        "bio_tag_counts": dict(tag_counts.most_common()),
        "fraction_O_tokens": tag_counts["O"] / n_tokens,
        "split_sizes": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "split_entities": split_ent,
        "cv_fold_test_sizes": [len(f["test"]) for f in folds],
        "train_test_identical_texts": text_overlap,
        "alignment": {
            "dropped_unknown_label": log["dropped_unknown_label"],
            "dropped_empty": log["dropped_empty"],
            "dropped_overlap": log["dropped_overlap"],
            "dropped_overlap_by_type": dict(log["dropped_overlap_by_type"]),
            "span_misaligned_expanded_to_word_boundary": log["span_misaligned_expanded"],
            "span_no_tokens": log["span_no_tokens"],
            "spans_after_overlap_resolution": log["orig_total"],
            "roundtrip_recovered_pct": 100 * log["roundtrip_recovered"] / log["roundtrip_total"],
            "original_span_exact_recovered_pct": 100 * log["orig_exact_recovered"] / log["orig_total"],
        },
    }
    save_json(RESULTS / "data_stats.json", stats)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    types = [t for t, _ in ent_counts.most_common()]
    axes[0].barh(types[::-1], [ent_counts[t] for t in types[::-1]], color="#4C72B0")
    axes[0].set_title("Entity spans per type")
    axes[0].set_xlabel("count")
    tok_types = sorted(ENTITY_TYPES, key=lambda t: -(tag_counts[f"B-{t}"] + tag_counts[f"I-{t}"]))
    axes[1].barh(tok_types[::-1], [tag_counts[f"B-{t}"] + tag_counts[f"I-{t}"] for t in tok_types[::-1]],
                 color="#DD8452")
    axes[1].set_xscale("log")
    axes[1].set_title(f"Word tokens per type (log); O = {tag_counts['O']:,}")
    axes[1].set_xlabel("tokens (log scale)")
    plt.tight_layout()
    plt.savefig(FIGURES / "label_distribution.png", dpi=150)
    print(json.dumps({k: v for k, v in stats.items() if k != "duplicate_pairs_(orig_index,kept_index)"}, indent=2))


if __name__ == "__main__":
    main()
