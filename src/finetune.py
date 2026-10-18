"""Phase 2B (AFTER) + Phase 3 ablations: fine-tune dslim/bert-base-NER on resume NER.

The CoNLL classification head (9 labels) is replaced by a new head for the 21 resume BIO labels
(ignore_mismatched_sizes=True); the BERT encoder weights are reused. Only the first sub-token of
each word is labelled (others -100). Long resumes are split into overlapping windows.

Examples
  python src/finetune.py --tag main --lr 5e-5 --epochs 10 --save_model
  python src/finetune.py --tag cw --class_weights
  python src/finetune.py --tag cv --cv            # 5-fold cross-validation
"""
import argparse
import copy
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import (BASE_MODEL, ENTITY_TYPES, ID2LABEL, LABEL2ID, LABELS, MODELS, RESULTS, load_split,
                    save_json, set_seed, write_jsonl)
from evaluate import entity_metrics, repair_bio, summarize, token_confusion
from inference import predict_word_probs, probs_to_tags
from rules import apply_rules


def encode_windows(docs, tok, max_length, stride, sliding):
    ids, masks, labels = [], [], []
    for d in docs:
        enc = tok(d["tokens"], is_split_into_words=True, truncation=True, max_length=max_length,
                  stride=stride if sliding else 0, return_overflowing_tokens=sliding,
                  padding="max_length")
        n = len(enc["input_ids"]) if sliding else 1
        for w in range(n):
            input_ids = enc["input_ids"][w] if sliding else enc["input_ids"]
            mask = enc["attention_mask"][w] if sliding else enc["attention_mask"]
            word_ids = enc.word_ids(w)
            lab, prev = [], None
            for wid in word_ids:
                lab.append(-100 if wid is None or wid == prev else LABEL2ID[d["tags"][wid]])
                prev = wid
            ids.append(input_ids); masks.append(mask); labels.append(lab)
    return TensorDataset(torch.tensor(ids), torch.tensor(masks), torch.tensor(labels))


def class_weight_vector(docs):
    """Inverse square-root label frequency, normalised to mean 1 over labels."""
    c = Counter(t for d in docs for t in d["tags"])
    w = np.array([1.0 / np.sqrt(max(c[l], 1)) for l in LABELS])
    return torch.tensor(w / w.mean(), dtype=torch.float)


def oversample(docs, factor=2):
    """Repeat resumes that contain a rare entity type (fewer spans than the median type)."""
    c = Counter(t for d in docs for _, _, t in d["spans"])
    med = np.median([c[t] for t in ENTITY_TYPES])
    rare = {t for t in ENTITY_TYPES if c[t] < med}
    extra = [d for d in docs if any(t in rare for _, _, t in d["spans"])]
    return docs + extra * (factor - 1), sorted(rare)


def predict_docs(model, tok, docs, args, device, rules=False):
    preds = []
    for d in docs:
        probs = predict_word_probs(model, tok, d["tokens"], args.max_length, args.stride,
                                   sliding=not args.truncate, device=device)
        tags = repair_bio(probs_to_tags(probs, ID2LABEL))
        if rules:
            tags = repair_bio(apply_rules(d["text"], d["starts"], d["ends"], tags))
        preds.append(tags)
    return preds


@torch.no_grad()
def eval_loss(model, loader, device, loss_fn):
    model.eval()
    tot, n = 0.0, 0
    for ids, mask, lab in loader:
        ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits
        tot += loss_fn(logits.float().view(-1, len(LABELS)), lab.view(-1)).item() * ids.size(0)
        n += ids.size(0)
    return tot / max(n, 1)


def train_one(split, args, device, log_prefix=""):
    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True).to(device)

    train_docs = split["train"]
    rare = None
    if args.oversample:
        train_docs, rare = oversample(train_docs)
    sliding = not args.truncate
    train_ds = encode_windows(train_docs, tok, args.max_length, args.stride, sliding)
    val_ds = encode_windows(split["val"], tok, args.max_length, args.stride, sliding)
    micro_bs = min(args.batch_size, args.micro_batch)
    accum = args.batch_size // micro_bs
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=micro_bs, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=micro_bs)

    weight = class_weight_vector(split["train"]).to(device) if args.class_weights else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
    plain_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = (len(train_loader) // accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)

    history, best_f1, best_state, best_epoch = [], -1.0, None, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for step, (ids, mask, lab) in enumerate(train_loader):
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float().view(-1, len(LABELS)), lab.view(-1))
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            tot += plain_loss(logits.detach().float().view(-1, len(LABELS)), lab.view(-1)).item() * ids.size(0)
            n += ids.size(0)
        model.eval()
        val_loss = eval_loss(model, val_loader, device, plain_loss)
        val_m = entity_metrics([d["tags"] for d in split["val"]], predict_docs(model, tok, split["val"], args, device))
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss,
                        "val_micro_f1": val_m["micro"]["f1"], "val_macro_f1": val_m["macro"]["f1"]})
        print(f"{log_prefix}epoch {epoch}: train_loss {tot / n:.4f} val_loss {val_loss:.4f} val {summarize(val_m)}",
              flush=True)
        if val_m["micro"]["f1"] > best_f1:
            best_f1, best_epoch = val_m["micro"]["f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t_start
    model.load_state_dict(best_state)
    model.eval()
    return model, tok, {"history": history, "best_epoch": best_epoch, "best_val_micro_f1": best_f1,
                        "train_time_sec": train_time, "n_train_windows": len(train_ds),
                        "n_train_docs": len(train_docs), "oversampled_rare_types": rare}


def test_eval(model, tok, docs, args, device):
    gold = [d["tags"] for d in docs]
    pred = predict_docs(model, tok, docs, args, device)
    pred_r = predict_docs(model, tok, docs, args, device, rules=True)
    return {"test": entity_metrics(gold, pred), "test_with_rules": entity_metrics(gold, pred_r),
            "confusion": token_confusion(gold, pred)}, pred, pred_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--micro_batch", type=int, default=8, help="per-step batch; grad accumulation fills batch_size")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--truncate", action="store_true", help="plain truncation instead of sliding windows")
    ap.add_argument("--class_weights", action="store_true")
    ap.add_argument("--oversample", action="store_true")
    ap.add_argument("--variant", default="clean", choices=["clean", "raw"], help="raw = no text normalisation")
    ap.add_argument("--cv", action="store_true", help="run 5-fold cross-validation")
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, folds = load_split(args.variant)
    config = vars(args) | {"base_model": BASE_MODEL, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                           "labels": LABELS, "precision": "bf16 autocast" if device == "cuda" else "fp32"}
    out_dir = RESULTS / "runs"

    if args.cv:
        fold_res = []
        for k, f in enumerate(folds):
            model, tok, info = train_one(f, args, device, log_prefix=f"[fold {k}] ")
            m, _, _ = test_eval(model, tok, f["test"], args, device)
            m.pop("confusion")
            fold_res.append(info | m)
            print(f"[fold {k}] test {summarize(m['test'])} | +rules {summarize(m['test_with_rules'])}", flush=True)
            del model; torch.cuda.empty_cache()
        summary = {}
        for key in ("test", "test_with_rules"):
            for avg in ("micro", "macro"):
                v = [r[key][avg]["f1"] for r in fold_res]
                summary[f"{key}_{avg}_f1"] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": v}
            summary[f"{key}_per_type_f1"] = {
                t: {"mean": float(np.mean([r[key]["per_type"][t]["f1"] for r in fold_res])),
                    "std": float(np.std([r[key]["per_type"][t]["f1"] for r in fold_res]))} for t in ENTITY_TYPES}
        save_json(out_dir / f"{args.tag}.json", {"config": config, "folds": fold_res, "summary": summary})
        print("CV micro-F1 %.4f ± %.4f" % (summary["test_micro_f1"]["mean"], summary["test_micro_f1"]["std"]))
        return

    model, tok, info = train_one(splits, args, device)
    m, pred, pred_r = test_eval(model, tok, splits["test"], args, device)
    res = {"config": config} | info | m
    if args.save_model:
        path = MODELS / args.tag
        model.save_pretrained(path)
        tok.save_pretrained(path)
        res["model_dir"] = str(path.relative_to(path.parents[1]))
        res["model_size_mb"] = sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 2**20
        write_jsonl(RESULTS / "predictions" / f"{args.tag}_test.jsonl",
                    [{"id": d["id"], "tags": p, "tags_rules": pr} for d, p, pr in zip(splits["test"], pred, pred_r)])
    save_json(out_dir / f"{args.tag}.json", res)
    print(f"[{args.tag}] best epoch {info['best_epoch']} | test {summarize(m['test'])} | "
          f"+rules {summarize(m['test_with_rules'])} | train {info['train_time_sec']:.0f}s")


if __name__ == "__main__":
    main()
