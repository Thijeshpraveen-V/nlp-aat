"""FastAPI backend for the resume parser + React demo.

    uvicorn app.api:app --port 8000
Endpoints
  POST /parse          multipart file (PDF/DOCX/TXT), form fields: model=finetuned|pretrained|both, rules=true|false
  POST /parse_text     JSON {"text": ..., "model": ..., "rules": ...}
  GET  /samples        sample resumes (with gold annotations when they come from the test split)
  GET  /samples/{name}
  GET  /metrics        before-vs-after summary read from results/*.json
  GET  /health
The built React app (frontend/dist) is served at /.
"""
import json
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import ENTITY_TYPES, RESULTS  # noqa: E402
from parse_resume import MODEL_NAMES, extract_text, parse_text  # noqa: E402

SAMPLES = ROOT / "data" / "sample_resumes"
DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="Resume NER API", version="1.0",
              description="Resume parsing with pretrained vs fine-tuned BERT NER")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def run(text, model, rules):
    names = MODEL_NAMES if model == "both" else (model,)
    if any(n not in MODEL_NAMES for n in names):
        raise HTTPException(400, f"model must be one of {MODEL_NAMES + ('both',)}")
    try:
        res = {n: parse_text(text, n, rules) for n in names}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return res[model] if model != "both" else {"results": res}


class TextIn(BaseModel):
    text: str
    model: str = "finetuned"
    rules: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "entity_types": ENTITY_TYPES}


@app.post("/parse")
async def parse(file: UploadFile = File(...), model: str = Form("finetuned"), rules: bool = Form(True)):
    data = await file.read()
    if len(data) > 10 * 2**20:
        raise HTTPException(413, "File larger than 10 MB")
    try:
        text = extract_text(data, file.filename or "")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return run(text, model, rules)


@app.post("/parse_text")
def parse_text_ep(body: TextIn):
    if not body.text.strip():
        raise HTTPException(422, "Empty text")
    return run(body.text, body.model, body.rules)


@app.get("/samples")
def samples():
    out = []
    for p in sorted(SAMPLES.glob("*")):
        if p.suffix.lower() in (".txt", ".pdf", ".docx"):
            out.append({"name": p.name, "has_gold": p.with_suffix(".gold.json").exists()})
    return out


@app.get("/samples/{name}")
def sample(name: str):
    p = (SAMPLES / name).resolve()
    if p.parent != SAMPLES.resolve() or not p.exists():
        raise HTTPException(404, "sample not found")
    text = extract_text(p.read_bytes(), p.name)
    gold = p.with_suffix(".gold.json")
    return {"name": name, "text": text, "gold": json.loads(gold.read_text("utf-8")) if gold.exists() else None}


@app.get("/metrics")
def metrics():
    p = RESULTS / "summary.json"
    if not p.exists():
        raise HTTPException(404, "Run src/evaluate.py first to build results/summary.json")
    return json.loads(p.read_text("utf-8"))


(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=RESULTS / "figures"), name="figures")

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(DIST / "index.html")
