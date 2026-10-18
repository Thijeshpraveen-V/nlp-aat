# Resume NER: Pretrained vs Fine-Tuned BERT — NLP for LLM AAT (24AM7PCLLM)

> Give this file to Claude Code together with `LLM_AAT_Report_Template.pdf` and `LLM_AAT_template.pptx`.
> Rename it to `CLAUDE.md` in the repo root if you want it loaded automatically.

## 1. Goal

Build a **resume parsing system using Named Entity Recognition (NER)** and show, with real measurements, how much **fine-tuning** improves an existing pretrained NER model on resumes.

**Core experiment (one model, two conditions):**

| Condition | Description |
|---|---|
| **Before: pretrained, normal performance** | `dslim/bert-base-NER` used as-is on resumes, no training |
| **After: fine-tuned performance** | The same checkpoint fine-tuned on a resume NER dataset, evaluated on the same test data |

**Deliverables**
1. Working code and a small real-time demo (web app and/or API)
2. A **report** following the college template section by section (DOCX and PDF)
3. A **presentation** following the college PPT template (PPTX)

**Assignment coverage**
- Part 1 (CO1): topic exploration, real-world application/API, architecture study, preliminary report
- Part 2: preprocessing, training/testing, performance enhancement, evaluation with proper metrics, before-vs-after comparison, report and PPT

## 2. Ground rules for Claude Code

- **Never fabricate results.** Every number, table and figure in the report must come from an actual run. Save metrics to `results/*.json` and generate tables and figures from those files by script.
- **Never invent references.** Cite only real papers and documentation, and verify names and years.
- Fix all random seeds (`42`) and log library versions to `results/environment.txt`.
- Training must run on **Google Colab or Kaggle free GPU**. Inference and the demo must run on a normal laptop CPU.
- Ask me for team details (names, USNs, section) instead of guessing. Use placeholders like `TODO_NAME_1` until I answer.
- After each phase, run its checks, then summarise what was done and the actual numbers.

## 3. Task definition

- **Input:** raw resume text (from PDF, DOCX or plain text)
- **Output:** a list of entities, each with type, text and character span
- **Tagging scheme:** BIO (`B-SKILLS`, `I-SKILLS`, `O`)
- **Target entity types** (final set depends on the dataset, roughly 8 to 10):
  `NAME`, `EMAIL`, `LOCATION`, `COLLEGE_NAME`, `DEGREE`, `GRADUATION_YEAR`, `COMPANIES_WORKED_AT`, `DESIGNATION`, `YEARS_OF_EXPERIENCE`, `SKILLS`

## 4. Dataset

- **Primary:** the Kaggle "Resume Entities for NER" dataset (DataTurks), about 200+ annotated resumes in JSON with character-span annotations. **Verify it is still available** before building on it.
- **Fallback:** another resume NER dataset from Kaggle or Hugging Face with span or BIO labels.
- Record in the report: exact source and URL, access date, licence, number of resumes, tokens, entities per type, class imbalance.

**Things Claude Code must handle**
- The dataset is small, so use a fixed **70/15/15 train/val/test split** and also **5-fold cross-validation** for the final fine-tuned numbers (mean ± std).
- Annotations are character spans. Convert them carefully to word-level BIO labels. Log how many spans fail to align to token boundaries and how they were handled.
- Check for duplicate resumes and leakage between splits.
- Produce a label-distribution plot.

## 5. Repository structure

```
resume-ner/
├── README.md
├── PROJECT_SPEC.md
├── requirements.txt
├── data/
│   ├── raw/                      # downloaded dataset
│   ├── processed/                # BIO files and splits
│   └── sample_resumes/           # 3 to 5 sample PDFs/TXTs for the demo
├── src/
│   ├── data_prep.py              # load, clean, span→BIO, splits, stats
│   ├── pretrained_eval.py        # BEFORE: run dslim/bert-base-NER as-is, map labels, evaluate
│   ├── finetune.py               # AFTER: fine-tune on resume data
│   ├── evaluate.py               # seqeval metrics, per-entity tables, CV aggregation
│   ├── error_analysis.py         # error categories and examples
│   └── parse_resume.py           # PDF/DOCX → text → entities (shared by demo and API)
├── notebooks/
│   └── train_colab.ipynb         # GPU fine-tuning run
├── app/
│   ├── api.py                    # FastAPI: POST /parse
│   └── demo.py                   # Streamlit UI
├── results/
│   ├── metrics_*.json
│   ├── figures/
│   └── environment.txt
├── report/
│   └── report.docx / report.pdf
└── slides/
    └── AAT_presentation.pptx
```

## 6. Implementation phases

### Phase 1: Setup and data
1. Create a virtual environment. `requirements.txt`: `transformers`, `datasets`, `torch`, `accelerate`, `seqeval`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `pdfplumber`, `python-docx`, `fastapi`, `uvicorn`, `streamlit`, `python-pptx`.
2. Write `data_prep.py` to:
   - load the JSON and clean the text (remove control characters, normalise whitespace while **preserving character offsets**)
   - split text into words with offsets (for example regex `\w+|[^\w\s]`) and convert spans to BIO
   - create the train/val/test split and the 5-fold CV indices, saved to disk
   - output dataset statistics
3. **Check:** convert BIO back to spans and confirm the original spans are recovered. Report the percentage recovered.

### Phase 2: Pretrained vs fine-tuned model

**Model:** `dslim/bert-base-NER`, a BERT-base model already fine-tuned for NER on CoNLL-2003 (labels PER, ORG, LOC, MISC).

**2A. Before: pretrained, normal performance (`pretrained_eval.py`)**
- Run the model as-is on the resume **test split**, with no training.
- Map its labels to ours: `PER→NAME`, `ORG→COMPANIES_WORKED_AT`, `LOC→LOCATION`. `MISC` is unmapped.
- Resume entity types it cannot predict (SKILLS, DEGREE, DESIGNATION, EMAIL, GRADUATION_YEAR, YEARS_OF_EXPERIENCE, COLLEGE_NAME) score 0 by construction. Report this honestly.
- Report **two views** so the comparison is fair:
  1. All entity types (full resume label set)
  2. Overlapping types only (NAME, COMPANIES_WORKED_AT, LOCATION)
- Also report an alternate mapping where `ORG` can match either `COMPANIES_WORKED_AT` or `COLLEGE_NAME`, since colleges are organisations too.
- Handle resumes longer than 512 tokens with sliding windows (stride about 128) and merge predictions.

**2B. After: fine-tuned performance (`finetune.py`)**
- Load the same checkpoint with a new head for our labels:
  `AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", num_labels=N, id2label=..., label2id=..., ignore_mismatched_sizes=True)`
- Subword alignment: label the first sub-token of each word, set the others to `-100`, and convert predictions back to word level.
- Same sliding-window handling for long resumes.
- Training configuration to log: learning rate (2e-5 / 3e-5 / 5e-5), batch size (8 / 16), epochs (5 to 10), weight decay 0.01, warmup 10%, max length 256 or 512, seed 42. Keep the best validation-F1 checkpoint.
- Evaluate on the **same test split** with the **same metric** as 2A.

### Phase 3: Performance enhancement (Part 2, task 2)
Run each of these on the fine-tuned model and report each as a separate row (ablation):
- Learning rate and epochs tuning
- Class-weighted loss (or oversampling resumes that contain rare entities)
- Sliding window vs plain truncation for long resumes
- Regex rule layer for EMAIL, phone and year patterns on top of the model
- Text normalisation vs none

### Phase 4: Evaluation
- **Entity-level** precision, recall and F1 with `seqeval` (strict mode), micro and macro averages.
- **Before vs After table** per entity type with columns: Before, After, Delta. Include overall rows.
- Grouped bar chart of F1 per entity type, Before vs After.
- Token-level confusion matrix for the fine-tuned model.
- Training and validation loss curves, and validation F1 per epoch.
- 5-fold CV mean ± std for the fine-tuned model. The pretrained model needs no training, so evaluate it on each fold's test part.
- Inference time per resume on CPU, model size, and training time.
- Save everything to `results/metrics_*.json` and `results/figures/`.

### Phase 5: Qualitative and error analysis
- Show **before vs after predictions on the same 8 to 10 test resume snippets** (text, gold labels, pretrained output, fine-tuned output).
- Categorise remaining errors of the fine-tuned model: boundary errors, type errors, missed entities, spurious entities.
- Identify the hardest entity types and explain why (for example SKILLS lists, designation vs company ambiguity, PDF layout noise).

### Phase 6: Real-time application / API
- `parse_resume.py`: accepts PDF, DOCX or TXT, extracts text with `pdfplumber`, runs the fine-tuned model, returns structured JSON.
- `api.py`: FastAPI `POST /parse` (file upload) returning `{ "name", "email", "skills": [], "education": [], "companies": [], "entities": [{"text","label","start","end"}] }`.
- `demo.py`: Streamlit UI with file upload, highlighted entities (colour per label), a structured table, and a **toggle to switch between the pretrained and fine-tuned model** so the before-vs-after difference can be shown live.
- Document I/O behaviour, latency and failure cases (scanned PDFs, multi-column layouts, tables) for report section 3.
- Real-world context for section 3: ATS platforms and commercial resume-parsing APIs. Describe them only from public documentation and cite it.

### Phase 7: Report (section 7 of this file)
### Phase 8: Presentation (section 8 of this file)

## 7. Report: follow the college template exactly

Use `LLM_AAT_Report_Template.pdf` for structure, including the cover page and table of contents.

**Cover page:** Topic: "Named Entity Recognition for Resume Parsing: Pretrained vs Fine-Tuned BERT". Fill team names, USNs, semester and section. Faculty in-charge is already printed.

**Template issues to fix in our copy:** the footer says `24AM4PCLLM` but the course code is `24AM7PCLLM`, and page 3 shows "Page 3 of 2". Use the correct code and automatic page numbering.

| Section | Content | Source |
|---|---|---|
| **1. Introduction** (1.1 to 1.4) | NLP background, why NER matters for hiring and ATS, objectives (measure the gain from fine-tuning, build a demo), scope | Written |
| **2. Selected NLP Topic Overview** (2.1 to 2.4) | NER task, BIO tagging, use cases, syllabus relevance, challenges (domain shift, ambiguity, layout noise, small data) | Written |
| **3. Real-Time Application / API** (3.1 to 3.5) | Our resume-parsing API and demo, real-world ATS context, architecture (PDF→text→NER→JSON), I/O, limitations | Phase 6 |
| **4. NLP Model / Architecture Study** (4.1 to 4.6) | Transformers and self-attention, BERT architecture, WordPiece subwords, token-classification head, pretraining vs fine-tuning, how `dslim/bert-base-NER` was built, equations (attention, softmax, cross-entropy loss), why BERT suits NER | Written and diagrams |
| **5. Dataset and Preprocessing** (5.1 to 5.5) | Dataset source and stats, cleaning, tokenisation (explain why stopwords are **not** removed for NER), BIO conversion, split and CV strategy, BERT subword embeddings | Phase 1 |
| **6. Implementation Methodology** (6.1 to 6.5) | Tools and versions, steps for 2A and 2B, API integration, pseudocode and workflow diagram, hyperparameter table | Phases 2, 3, 6 |
| **7. Results and Evaluation** (7.1 to 7.5) | Metrics tables, before-vs-after table and chart, confusion matrix, loss curves, qualitative examples, error analysis, ablations. 7.5 is the pretrained vs fine-tuned comparison | Phases 4, 5 |
| **8. Discussion** (8.1 to 8.5) | Why fine-tuning helps (label-set and domain shift), strengths and weaknesses, challenges (span alignment, tiny data, long resumes), effect of preprocessing and data quality, how the API helps | Written from real results |
| **9. Conclusion** (9.1 to 9.4) | Summary, learnings, real-world applicability, future work (layout-aware models such as LayoutLM, larger datasets, LLM-based extraction, fairness and bias in hiring) | Written |
| **10. References** | IEEE style, real sources only | Section 10 below |

Notes:
- Produce DOCX with proper heading styles (auto-generated TOC), then export to PDF.
- Number and caption every figure and table, and refer to each in the text.
- Target length: roughly 20 to 30 pages including figures.

## 8. Presentation: follow `LLM_AAT_template.pptx`

Inspect the template's layouts and placeholders first. **Reuse its theme, fonts and slide structure**, and keep any predefined slide titles in order. Suggested flow, adapted to the template (about 12 to 15 slides, mostly visuals, speaker notes included):

1. Title and team
2. Problem and motivation (ATS, resume volume)
3. NER task and entity types, with a highlighted resume example
4. Application / API and demo architecture
5. Dataset overview and label distribution
6. Preprocessing pipeline
7. BERT architecture and the pretrained vs fine-tuned idea
8. Training setup and hyperparameters
9. **Before vs after results** (table and chart)
10. Qualitative examples (same snippet, both models)
11. Ablations and error analysis
12. Live demo screenshots (be ready for a live demo with the model toggle)
13. Challenges, conclusion, future work
14. References

## 9. Final checklist

- [ ] `requirements.txt` works from a clean environment
- [ ] `README.md` has the exact commands to reproduce the before and after results
- [ ] All metrics JSON and figures are generated by scripts
- [ ] Demo runs: `streamlit run app/demo.py` and `uvicorn app.api:app`
- [ ] Report DOCX and PDF follow the template headings, numbering and cover page
- [ ] PPTX uses the college template
- [ ] Every number in the report matches a results file
- [ ] Team names, USNs and section are filled in on the cover page

## 10. Reference starting points (verify each before citing)

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- Tjong Kim Sang and De Meulder, "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition", 2003 (training data of the pretrained model)
- Model card for `dslim/bert-base-NER` on Hugging Face
- Hugging Face Transformers documentation, token classification guide
- `seqeval` library documentation
- Dataset page (Kaggle or Hugging Face) with exact URL and access date
- Public documentation of any commercial resume-parsing API or ATS mentioned in section 3

## 11. Suggested order of work

1. Phase 1 (data) and 2A (pretrained baseline on CPU) to get the "before" numbers first
2. Phase 2B (fine-tuning) on a Colab GPU
3. Phase 3 ablations and CV runs
4. Phases 4 and 5 (evaluation, error analysis, figures)
5. Phase 6 demo and screenshots
6. Report and PPT, then a final consistency pass so the numbers in the text match the tables
