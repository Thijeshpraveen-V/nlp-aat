"""Word-level inference for any token-classification checkpoint, with sliding windows over
long resumes (or plain truncation for the ablation). Predictions of a word are taken from its
first sub-token; where windows overlap, softmax probabilities are averaged."""
import numpy as np
import torch


@torch.no_grad()
def predict_word_probs(model, tokenizer, words, max_length=512, stride=128, sliding=True,
                       device="cpu", batch_size=8):
    """Return array [n_words, n_labels] of probabilities. Words not covered (truncation mode)
    get NaN rows."""
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length,
                    stride=stride if sliding else 0, return_overflowing_tokens=sliding,
                    padding=True, return_tensors="pt")
    n_windows = enc["input_ids"].shape[0]
    word_ids = [enc.word_ids(i) for i in range(n_windows)]
    n_labels = model.config.num_labels
    acc = np.zeros((len(words), n_labels), dtype=np.float64)
    cnt = np.zeros(len(words))
    for b in range(0, n_windows, batch_size):
        inputs = {k: enc[k][b:b + batch_size].to(device) for k in ("input_ids", "attention_mask")}
        if "token_type_ids" in enc:
            inputs["token_type_ids"] = enc["token_type_ids"][b:b + batch_size].to(device)
        probs = torch.softmax(model(**inputs).logits.float(), -1).cpu().numpy()
        for j in range(probs.shape[0]):
            prev = None
            for k, w in enumerate(word_ids[b + j]):
                if w is not None and w != prev:  # first sub-token of the word
                    acc[w] += probs[j, k]
                    cnt[w] += 1
                prev = w
    out = np.full_like(acc, np.nan)
    covered = cnt > 0
    out[covered] = acc[covered] / cnt[covered, None]
    return out


def probs_to_tags(probs, id2label):
    tags = []
    for row in probs:
        tags.append("O" if np.isnan(row).any() else id2label[int(row.argmax())])
    return tags
