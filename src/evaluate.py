"""Metrics (seqeval, strict IOB2) shared by all experiments, plus aggregation of saved runs
into the final tables and figures (run as a script after the experiments)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
from collections import Counter

import numpy as np
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from common import ENTITY_TYPES


def repair_bio(tags):
    """Make a tag sequence valid IOB2: an I-X that does not follow B-X/I-X becomes B-X."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


def restrict(tags, types):
    return [t if t != "O" and t[2:] in types else "O" for t in tags]


def entity_metrics(gold, pred, types=None):
    """Entity-level P/R/F1 (seqeval strict, IOB2). `types` restricts evaluation to a subset
    (other gold and predicted types are set to O). Per-type rows are returned for every
    type in `types` (0 if the model never predicts it)."""
    types = list(types or ENTITY_TYPES)
    g = [restrict(s, types) for s in gold]
    p = [restrict(repair_bio(s), types) for s in pred]
    rep = classification_report(g, p, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
    per_type = {}
    for t in types:
        r = rep.get(t, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        per_type[t] = {"precision": float(r["precision"]), "recall": float(r["recall"]),
                       "f1": float(r["f1-score"]), "support": int(r["support"])}
    present = [t for t in types if per_type[t]["support"] > 0]
    micro = rep["micro avg"]
    return {
        "types": types,
        "per_type": per_type,
        "micro": {"precision": float(micro["precision"]), "recall": float(micro["recall"]),
                  "f1": float(micro["f1-score"]), "support": int(micro["support"])},
        "macro": {k: float(np.mean([per_type[t][k] for t in present])) for k in ("precision", "recall", "f1")},
    }


def token_confusion(gold, pred, types=None):
    """Word-level confusion matrix on entity *types* (B/I collapsed), O included."""
    types = list(types or ENTITY_TYPES)
    labels = ["O"] + types
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for gs, ps in zip(gold, pred):
        for g, p in zip(gs, ps):
            gt = "O" if g == "O" else g[2:]
            pt = "O" if p == "O" else p[2:]
            m[idx.get(gt, 0), idx.get(pt, 0)] += 1
    return {"labels": labels, "matrix": m.tolist()}


def summarize(metrics):
    return f"micro-F1 {metrics['micro']['f1']:.4f} | macro-F1 {metrics['macro']['f1']:.4f}"


# ---------------------------------------------------------------------------------------------
# Aggregation: python src/evaluate.py  (after pretrained_eval, finetune runs, benchmark)
# ---------------------------------------------------------------------------------------------
OVERLAP = ["NAME", "COMPANIES_WORKED_AT", "LOCATION"]
ABLATIONS = [  # (run tag, label) ; rules row is added from the final run
    ("abl_ep5", "Epochs = 5 (instead of 10)"),
    ("abl_cw", "+ class-weighted loss"),
    ("abl_os", "+ oversampling rare-entity resumes"),
    ("abl_trunc", "Plain truncation (512) instead of sliding window"),
    ("abl_raw", "No text normalisation"),
]


def _fig_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    return plt, sns


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def aggregate():
    from common import FIGURES, RESULTS, load_json, load_split, read_jsonl, save_json

    plt, sns = _fig_setup()
    tables = RESULTS / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    splits, _ = load_split("clean")
    test = splits["test"]
    gold = [d["tags"] for d in test]
    final_tag = load_json(RESULTS / "final_config.json")["final_run"]
    final = load_json(RESULTS / "runs" / f"{final_tag}.json")
    pre = load_json(RESULTS / "metrics_pretrained.json")

    pre_pred = {r["id"]: r["tags"] for r in read_jsonl(RESULTS / "predictions" / "pretrained_test.jsonl")}
    ft = {r["id"]: r for r in read_jsonl(RESULTS / "predictions" / f"{final_tag}_test.jsonl")}
    P = [pre_pred[d["id"]] for d in test]
    F = [ft[d["id"]]["tags"] for d in test]
    FR = [ft[d["id"]]["tags_rules"] for d in test]

    views = {
        "all_types": (entity_metrics(gold, P), entity_metrics(gold, F), entity_metrics(gold, FR)),
        "overlap_types": (entity_metrics(gold, P, OVERLAP), entity_metrics(gold, F, OVERLAP),
                          entity_metrics(gold, FR, OVERLAP)),
    }
    before, after, after_rules = views["all_types"]

    # ---- before vs after table
    rows = []
    for t in ENTITY_TYPES:
        b, a = before["per_type"][t], after["per_type"][t]
        rows.append({"type": t, "support": a["support"], "before_p": b["precision"], "before_r": b["recall"],
                     "before": b["f1"], "after_p": a["precision"], "after_r": a["recall"], "after": a["f1"],
                     "delta": a["f1"] - b["f1"]})
    overall = []
    for name, (b, a, _) in (("All types", views["all_types"]), ("Overlap types (NAME/COMPANY/LOC)", views["overlap_types"])):
        for avg in ("micro", "macro"):
            overall.append({"metric": f"{avg.capitalize()} F1 — {name}", "before": b[avg]["f1"], "after": a[avg]["f1"],
                            "delta": a[avg]["f1"] - b[avg]["f1"]})
    alt = pre["test"]["alt_all_types"]
    ba = {"rows": rows, "overall": overall, "views": {k: {"before": v[0], "after": v[1], "after_rules": v[2]}
                                                      for k, v in views.items()},
          "pretrained_alt_mapping_test": {"all_types": pre["test"]["alt_all_types"]["micro"] | {"macro_f1": alt["macro"]["f1"]},
                                          "overlap_types_plus_college": pre["test"]["alt_overlap_types"]["micro"]}}
    save_json(RESULTS / "metrics_before_after.json", ba)
    f = lambda v: f"{100 * v:.1f}"
    md = _md_table(["Entity type", "Support", "Before P", "Before R", "Before F1", "After P", "After R", "After F1", "Δ F1"],
                   [[r["type"], r["support"], f(r["before_p"]), f(r["before_r"]), f(r["before"]), f(r["after_p"]),
                     f(r["after_r"]), f(r["after"]), "+" + f(r["delta"])] for r in rows])
    md += "\n" + _md_table(["Overall", "Before", "After", "Δ"],
                           [[o["metric"], f(o["before"]), f(o["after"]), "+" + f(o["delta"])] for o in overall])
    (tables / "before_after.md").write_text(md, encoding="utf-8")

    # ---- CV
    cv = []
    for view in ("all_types", "overlap_types"):
        s = pre["cv_summary"][view]
        cv.append({"model": f"Pretrained ({view.replace('_', ' ')})", "micro_mean": s["micro_f1_mean"],
                   "micro_std": s["micro_f1_std"], "macro_mean": s["macro_f1_mean"], "macro_std": s["macro_f1_std"]})
    cv_run = RESULTS / "runs" / "cv_main.json"
    cv_data = load_json(cv_run) if cv_run.exists() else None
    if cv_data:
        for key, label in (("test", "Fine-tuned"), ("test_with_rules", "Fine-tuned + rules")):
            s = cv_data["summary"]
            cv.append({"model": f"{label} (all types)", "micro_mean": s[f"{key}_micro_f1"]["mean"],
                       "micro_std": s[f"{key}_micro_f1"]["std"], "macro_mean": s[f"{key}_macro_f1"]["mean"],
                       "macro_std": s[f"{key}_macro_f1"]["std"]})
    save_json(RESULTS / "metrics_cv.json", {"rows": cv, "fine_tuned_per_type": cv_data["summary"]["test_per_type_f1"] if cv_data else None})
    (tables / "cv.md").write_text(_md_table(["Model", "Micro F1 (mean ± std)", "Macro F1 (mean ± std)"],
                                            [[r["model"], f"{f(r['micro_mean'])} ± {f(r['micro_std'])}",
                                              f"{f(r['macro_mean'])} ± {f(r['macro_std'])}"] for r in cv]), encoding="utf-8")

    # ---- ablations + HP grid
    grid = load_json(RESULTS / "final_config.json")["grid"]
    abl = [{"name": f"Final fine-tuned (lr={final['config']['lr']}, bs={final['config']['batch_size']}, "
                    f"epochs={final['config']['epochs']}, sliding window)",
            "val_micro_f1": final["best_val_micro_f1"], "micro_f1": after["micro"]["f1"], "macro_f1": after["macro"]["f1"]},
           {"name": "+ regex rule layer (EMAIL, GRADUATION_YEAR)", "val_micro_f1": None,
            "micro_f1": after_rules["micro"]["f1"], "macro_f1": after_rules["macro"]["f1"]}]
    for tag, label in ABLATIONS:
        p = RESULTS / "runs" / f"{tag}.json"
        if p.exists():
            r = load_json(p)
            abl.append({"name": label, "val_micro_f1": r["best_val_micro_f1"], "micro_f1": r["test"]["micro"]["f1"],
                        "macro_f1": r["test"]["macro"]["f1"]})
    hp = [{"name": f"lr={g['lr']}, bs={g['batch_size']}", "val_micro_f1": g["best_val_micro_f1"],
           "best_epoch": g["best_epoch"], "micro_f1": g["test_micro_f1"]} for g in grid]
    save_json(RESULTS / "metrics_ablations.json", {"ablations": abl, "hp_grid": hp})
    fv = lambda v: "—" if v is None else f(v)
    (tables / "ablations.md").write_text(
        _md_table(["Setting", "Val micro F1", "Test micro F1", "Test macro F1"],
                  [[a["name"], fv(a["val_micro_f1"]), f(a["micro_f1"]), f(a["macro_f1"])] for a in abl])
        + "\n" + _md_table(["Grid run", "Val micro F1", "Best epoch", "Test micro F1"],
                           [[h["name"], f(h["val_micro_f1"]), h["best_epoch"], f(h["micro_f1"])] for h in hp]),
        encoding="utf-8")

    # ---- figures
    x = np.arange(len(ENTITY_TYPES))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.2, [r["before"] for r in rows], 0.4, label="Before: pretrained (mapped)", color="#9aa5b1")
    ax.bar(x + 0.2, [r["after"] for r in rows], 0.4, label="After: fine-tuned", color="#3b5bdb")
    ax.set_xticks(x, [t.replace("_", "\n") for t in ENTITY_TYPES], fontsize=8.5)
    ax.set_ylabel("Entity-level F1 (strict)")
    ax.set_ylim(0, 1)
    ax.set_title("F1 per entity type on the test split: before vs after fine-tuning")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "before_after_f1.png", dpi=160); plt.close()

    conf = final["confusion"]
    m = np.array(conf["matrix"], dtype=float)
    mn = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(mn, annot=True, fmt=".2f", cmap="Blues", xticklabels=conf["labels"], yticklabels=conf["labels"],
                cbar_kws={"label": "row-normalised"}, ax=ax, annot_kws={"size": 7.5})
    ax.set_xlabel("Predicted (word level)"); ax.set_ylabel("Gold (word level)")
    ax.set_title("Token-level confusion matrix, fine-tuned model (test split)")
    plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGURES / "confusion_matrix_finetuned.png", dpi=160); plt.close()

    h = final["history"]
    ep = [e["epoch"] for e in h]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ep, [e["train_loss"] for e in h], "o-", label="train loss")
    axes[0].plot(ep, [e["val_loss"] for e in h], "s-", label="validation loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(ep, [e["val_micro_f1"] for e in h], "o-", label="val micro F1")
    axes[1].plot(ep, [e["val_macro_f1"] for e in h], "s-", label="val macro F1")
    axes[1].axvline(final["best_epoch"], ls="--", c="gray", label=f"best epoch ({final['best_epoch']})")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("F1"); axes[1].legend(); axes[1].set_title("Validation F1")
    plt.tight_layout(); plt.savefig(FIGURES / "training_curves.png", dpi=160); plt.close()

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(abl) + 1))
    names = [a["name"] for a in abl][::-1]
    ax.barh(names, [a["micro_f1"] for a in abl][::-1], color="#3b5bdb")
    for i, v in enumerate([a["micro_f1"] for a in abl][::-1]):
        ax.text(v + 0.005, i, f"{100 * v:.1f}", va="center", fontsize=9)
    ax.set_xlabel("Test micro F1"); ax.set_title("Ablations (fine-tuned model)")
    ax.set_xlim(0, max(a["micro_f1"] for a in abl) * 1.15)
    plt.tight_layout(); plt.savefig(FIGURES / "ablations.png", dpi=160); plt.close()

    if cv_data:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        pre_f = [fo["all_types"]["micro"]["f1"] for fo in pre["cv_folds"]]
        ft_f = cv_data["summary"]["test_micro_f1"]["folds"]
        k = np.arange(1, len(ft_f) + 1)
        ax.bar(k - 0.2, pre_f, 0.4, label="Before: pretrained", color="#9aa5b1")
        ax.bar(k + 0.2, ft_f, 0.4, label="After: fine-tuned", color="#3b5bdb")
        ax.set_xticks(k, [f"fold {i}" for i in k]); ax.set_ylabel("Test micro F1"); ax.set_ylim(0, 1)
        ax.set_title("5-fold cross-validation"); ax.legend()
        plt.tight_layout(); plt.savefig(FIGURES / "cv_folds.png", dpi=160); plt.close()

    # ---- summary for the web app
    eff_p = RESULTS / "metrics_efficiency.json"
    summary = {
        "base_model": pre["model"], "final_run": final_tag, "n_test_resumes": len(test),
        "before_after": {"rows": rows, "overall": overall},
        "cv": cv, "ablations": abl,
        "efficiency": load_json(eff_p) if eff_p.exists() else None,
        "figures": [p.name for p in sorted(FIGURES.glob("*.png"))],
    }
    save_json(RESULTS / "summary.json", summary)
    print((tables / "before_after.md").read_text(encoding="utf-8"))
    print((tables / "cv.md").read_text(encoding="utf-8"))
    print((tables / "ablations.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    aggregate()
