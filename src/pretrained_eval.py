"""Phase 2A (BEFORE): run dslim/bert-base-NER as-is (no training) on resumes, map its CoNLL
labels to the resume label set and evaluate with the same metric as the fine-tuned model.

Views reported
  all_types          : all 10 resume entity types (types it cannot predict score 0)
  overlap_types      : NAME, COMPANIES_WORKED_AT, LOCATION only
  alt_*              : ORG may match COMPANIES_WORKED_AT or COLLEGE_NAME (each predicted ORG
                       entity takes whichever of the two the gold label at its first word is;
                       otherwise COMPANIES_WORKED_AT). This is an optimistic upper bound.
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common import BASE_MODEL, RESULTS, load_split, read_jsonl, DATA_PROCESSED, save_json, set_seed, write_jsonl
from evaluate import entity_metrics, repair_bio, summarize
from inference import predict_word_probs, probs_to_tags

CONLL_MAP = {"PER": "NAME", "ORG": "COMPANIES_WORKED_AT", "LOC": "LOCATION"}  # MISC unmapped
OVERLAP = ["NAME", "COMPANIES_WORKED_AT", "LOCATION"]
OVERLAP_ALT = OVERLAP + ["COLLEGE_NAME"]


def map_tags(conll_tags):
    out = []
    for t in conll_tags:
        if t == "O" or t[2:] not in CONLL_MAP:
            out.append("O")
        else:
            out.append(t[:2] + CONLL_MAP[t[2:]])
    return repair_bio(out)


def alt_map(conll_tags, gold):
    """ORG entity -> COLLEGE_NAME if the gold type at its first word is COLLEGE_NAME."""
    tags = repair_bio(["O" if t == "O" or t[2:] == "MISC" else t for t in conll_tags])
    out, cur = [], None
    for t, g in zip(tags, gold):
        if t == "O":
            out.append("O"); cur = None; continue
        typ = t[2:]
        if typ == "ORG":
            if t.startswith("B-"):
                cur = "COLLEGE_NAME" if g[2:] == "COLLEGE_NAME" else "COMPANIES_WORKED_AT"
            out.append(t[:2] + cur)
        else:
            out.append(t[:2] + CONLL_MAP[typ])
    return out


def evaluate_views(docs, conll_preds):
    gold = [d["tags"] for d in docs]
    pred = [map_tags(conll_preds[d["id"]]) for d in docs]
    alt = [alt_map(conll_preds[d["id"]], d["tags"]) for d in docs]
    return {
        "all_types": entity_metrics(gold, pred),
        "overlap_types": entity_metrics(gold, pred, OVERLAP),
        "alt_all_types": entity_metrics(gold, alt),
        "alt_overlap_types": entity_metrics(gold, alt, OVERLAP_ALT),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()
    set_seed()
    splits, folds = load_split("clean")
    all_docs = read_jsonl(DATA_PROCESSED / "clean" / "docs.jsonl")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL).eval()
    id2label = model.config.id2label

    # 1) CPU timing on the test split (the "laptop CPU" deployment setting)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    times = []
    conll_preds = {}
    predict_word_probs(model, tok, splits["train"][0]["tokens"], args.max_length, args.stride)  # warm-up
    for d in splits["test"]:
        t0 = time.perf_counter()
        probs = predict_word_probs(model, tok, d["tokens"], args.max_length, args.stride, device="cpu")
        times.append(time.perf_counter() - t0)
        conll_preds[d["id"]] = probs_to_tags(probs, id2label)

    # 2) remaining docs (for CV-fold evaluation) on GPU if available
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    for d in all_docs:
        if d["id"] not in conll_preds:
            probs = predict_word_probs(model, tok, d["tokens"], args.max_length, args.stride, device=dev)
            conll_preds[d["id"]] = probs_to_tags(probs, id2label)

    res = {
        "model": BASE_MODEL,
        "label_mapping": CONLL_MAP | {"MISC": None},
        "max_length": args.max_length, "stride": args.stride,
        "test": evaluate_views(splits["test"], conll_preds),
        "cv_folds": [evaluate_views(f["test"], conll_preds) for f in folds],
        "cpu_inference_sec_per_resume": {"mean": float(np.mean(times)), "std": float(np.std(times)),
                                         "n": len(times), "threads": torch.get_num_threads()},
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }
    for view in ("all_types", "overlap_types", "alt_all_types", "alt_overlap_types"):
        f1s = [f[view]["micro"]["f1"] for f in res["cv_folds"]]
        mf1s = [f[view]["macro"]["f1"] for f in res["cv_folds"]]
        res.setdefault("cv_summary", {})[view] = {
            "micro_f1_mean": float(np.mean(f1s)), "micro_f1_std": float(np.std(f1s)),
            "macro_f1_mean": float(np.mean(mf1s)), "macro_f1_std": float(np.std(mf1s))}
    save_json(RESULTS / "metrics_pretrained.json", res)
    write_jsonl(RESULTS / "predictions" / "pretrained_conll_raw.jsonl",
                [{"id": i, "tags": t} for i, t in conll_preds.items()])
    write_jsonl(RESULTS / "predictions" / "pretrained_test.jsonl",
                [{"id": d["id"], "tags": map_tags(conll_preds[d["id"]])} for d in splits["test"]])
    for view, m in res["test"].items():
        print(f"[test] {view:18s} {summarize(m)}")
    for view, s in res["cv_summary"].items():
        print(f"[cv]   {view:18s} micro-F1 {s['micro_f1_mean']:.4f} ± {s['micro_f1_std']:.4f}")
    print("CPU sec/resume", res["cpu_inference_sec_per_resume"])


if __name__ == "__main__":
    main()
