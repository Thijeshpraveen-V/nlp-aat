"""CPU inference benchmark for both models under identical conditions (test split, sliding
window 512/128). Models are timed alternately on each resume, 3 repeats, keeping the fastest
repeat per resume to reduce laptop timing noise. Writes results/metrics_efficiency.json.

Run in a foreground terminal: Windows power throttling (EcoQoS) slows down background processes."""
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common import BASE_MODEL, MODELS, RESULTS, load_json, load_split, save_json
from inference import predict_word_probs

REPEATS = 3


def main():
    torch.manual_seed(42)
    splits, _ = load_split("clean")
    docs = splits["test"]
    final = load_json(RESULTS / "final_config.json")["final_run"]
    paths = {"pretrained": BASE_MODEL, "finetuned": str(MODELS / final)}
    models = {k: (AutoTokenizer.from_pretrained(p), AutoModelForTokenClassification.from_pretrained(p).eval())
              for k, p in paths.items()}
    for tok, model in models.values():  # warm-up
        predict_word_probs(model, tok, docs[0]["tokens"])
    times = {k: [] for k in models}
    with torch.inference_mode():
        for d in docs:
            best = {k: float("inf") for k in models}
            for _ in range(REPEATS):
                for k, (tok, model) in models.items():
                    t0 = time.perf_counter()
                    predict_word_probs(model, tok, d["tokens"])
                    best[k] = min(best[k], time.perf_counter() - t0)
            for k in models:
                times[k].append(best[k])
    words = sum(len(d["tokens"]) for d in docs)
    res = {"device": "cpu", "threads": torch.get_num_threads(), "n_resumes": len(docs),
           "mean_words_per_resume": words / len(docs), "repeats_min_of": REPEATS}
    for k, (tok, model) in models.items():
        t = np.array(times[k])
        res[k] = {"sec_per_resume_mean": float(t.mean()), "sec_per_resume_median": float(np.median(t)),
                  "sec_per_resume_p95": float(np.percentile(t, 95)), "words_per_sec": float(words / t.sum()),
                  "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    res["finetuned"]["model_size_mb"] = sum(f.stat().st_size for f in (MODELS / final).glob("*.safetensors")) / 2**20
    run = load_json(RESULTS / "runs" / f"{final}.json")
    res["finetuned"]["train_time_sec"] = run["train_time_sec"]
    res["finetuned"]["train_device"] = run["config"]["device"]
    save_json(RESULTS / "metrics_efficiency.json", res)
    print(res)


if __name__ == "__main__":
    main()
