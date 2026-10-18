"""Shared helpers: label set, word tokenisation with offsets, BIO <-> span conversion, IO."""
import json
import os
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "Entity Recognition in Resumes.json"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"
SEED = 42
BASE_MODEL = "dslim/bert-base-NER"

# DataTurks label name -> our entity type
RAW2TYPE = {
    "Name": "NAME",
    "Email Address": "EMAIL",
    "Location": "LOCATION",
    "College Name": "COLLEGE_NAME",
    "Degree": "DEGREE",
    "Graduation Year": "GRADUATION_YEAR",
    "Companies worked at": "COMPANIES_WORKED_AT",
    "Designation": "DESIGNATION",
    "Years of Experience": "YEARS_OF_EXPERIENCE",
    "Skills": "SKILLS",
}
ENTITY_TYPES = list(RAW2TYPE.values())
LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

WORD_RE = re.compile(r"\w+|[^\w\s]")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def word_tokenize(text: str):
    """Split text into words with character offsets [start, end)."""
    toks, starts, ends = [], [], []
    for m in WORD_RE.finditer(text):
        toks.append(m.group())
        starts.append(m.start())
        ends.append(m.end())
    return toks, starts, ends


def bio_to_spans(tags, starts, ends):
    """Convert word-level BIO tags back to character spans [(start, end, type)].
    An I- tag without a matching preceding tag starts a new entity (lenient decoding)."""
    spans, cur = [], None
    for tag, s, e in zip(tags, starts, ends):
        if tag == "O":
            if cur:
                spans.append(tuple(cur))
            cur = None
            continue
        prefix, typ = tag.split("-", 1)
        if prefix == "B" or cur is None or cur[2] != typ:
            if cur:
                spans.append(tuple(cur))
            cur = [s, e, typ]
        else:
            cur[1] = e
    if cur:
        spans.append(tuple(cur))
    return spans


def spans_to_entities(text, spans):
    return [{"text": text[s:e], "label": t, "start": s, "end": e} for s, e, t in spans]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_split(variant="clean"):
    """Return dict split_name -> list of docs, plus the CV folds."""
    d = DATA_PROCESSED / variant
    docs = {r["id"]: r for r in read_jsonl(d / "docs.jsonl")}
    splits = load_json(DATA_PROCESSED / "splits.json")
    out = {k: [docs[i] for i in splits[k]] for k in ("train", "val", "test")}
    folds = [{k: [docs[i] for i in f[k]] for k in ("train", "val", "test")} for f in splits["folds"]]
    return out, folds
