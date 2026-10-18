"""Pick the fine-tuning configuration with the best VALIDATION micro-F1 from the hp_* grid runs
(test scores are not used for selection). Writes results/final_config.json and prints CLI args."""
from common import RESULTS, load_json, save_json

runs = []
for p in sorted((RESULTS / "runs").glob("hp_*.json")):
    r = load_json(p)
    runs.append({"tag": p.stem, "lr": r["config"]["lr"], "batch_size": r["config"]["batch_size"],
                 "epochs": r["config"]["epochs"], "best_epoch": r["best_epoch"],
                 "best_val_micro_f1": r["best_val_micro_f1"], "test_micro_f1": r["test"]["micro"]["f1"]})
best = max(runs, key=lambda r: r["best_val_micro_f1"])
save_json(RESULTS / "final_config.json", {"selected_by": "best validation micro-F1", "grid": runs,
                                          "best": best, "final_run": "main"})
for r in runs:
    print(f"{r['tag']:18s} val {r['best_val_micro_f1']:.4f} (epoch {r['best_epoch']}) test {r['test_micro_f1']:.4f}")
print(f"--lr {best['lr']} --batch_size {best['batch_size']} --epochs {best['epochs']}")
