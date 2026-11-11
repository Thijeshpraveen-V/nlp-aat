# Presentation Script: Resume NER, Pretrained vs Fine-Tuned BERT

**Speakers:** Thijesh Praveen (Slides 1–9) · Uday Kumar M (Slides 10–18 + live demo)
**Target length:** about 12–14 minutes of talking plus a 3–4 minute demo

---

## Before you present (checklist)

Do this at least 30 minutes before, on the laptop you will present from.

1. **Python environment** (one time only; the repo has no `.venv` yet):
   ```powershell
   cd C:\codes\nlp-aat
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   (CPU PyTorch is enough for the demo. You do not need the GPU.)
2. **Check the model is there:** `models\main\` must contain `model.safetensors`, `config.json` and `tokenizer.json`.
   It does on this machine. If you present from another laptop, copy the whole `models\main` folder over.
3. **Frontend build:** `frontend\dist\` is already built. Only if it is missing, run:
   ```powershell
   cd frontend; npm install; npm run build; cd ..
   ```
4. **Start the server** in a normal, visible terminal (not a background one; Windows throttles background processes and inference gets about 5x slower):
   ```powershell
   .venv\Scripts\python.exe -m uvicorn app.api:app --port 8000
   ```
   Wait for `Application startup complete`. The first parse loads the models, so run one sample before the talk to warm it up.
5. **Open these browser tabs in advance:**
   - `http://127.0.0.1:8000/?sample=test_resume_082.txt&mode=compare&view=text` (the side-by-side demo)
   - `http://127.0.0.1:8000/#results` (the Results tab)
   - `http://127.0.0.1:8000/docs` (Swagger API docs)
6. **Backup:** if anything fails live, Slide 14 has screenshots (`results/screenshots/`). Talk over those.

---

## Part 1: Thijesh (Slides 1–9)

### Slide 1: Title
**Thijesh:**
"Good morning. I'm Thijesh Praveen, and with me is Uday Kumar. Our AAT project for Natural Language Processing for LLM, course code 24AM7PCLLM, under Upeksha Prasad ma'am, is **Resume NER: Pretrained versus Fine-Tuned BERT**.
In short, we took a BERT model trained to find named entities in news articles and measured how well it works on resumes. Then we fine-tuned it on resumes and measured how much better it gets. I'll cover the problem, the dataset and the model. Uday will cover the results, the live demo and the conclusions."

### Slide 2: Agenda
**Thijesh:**
"Here's the flow: introduction, the NLP topic, the real-world application, the model architecture, the dataset and preprocessing, and our implementation. Then Uday takes over for results, discussion, conclusion and references."

### Slide 3: Introduction: Why Resume Parsing Needs NER
**Thijesh:**
"Companies receive resumes as PDFs or Word files, and no two have the same layout. An Applicant Tracking System needs structured fields: the candidate's name, contact details, education, employers, job titles and skills.
Named Entity Recognition is the NLP task that does this. It reads raw text and marks spans like 'this is a NAME' or 'this is a COMPANY', with exact character positions.
Our research question was: **how good is an off-the-shelf pretrained NER model on resumes, and how much does fine-tuning add?**
On the right are the headline numbers: 219 annotated resumes from the DataTurks Kaggle dataset, 10 resume entity types, and the main result. Test micro-F1 went from **10.3% before fine-tuning to 44.8% after**, more than four times higher. Besides the measured comparison, we also built a REST API and a live web demo, which Uday will show."

### Slide 4: Selected NLP Topic: Named Entity Recognition
**Thijesh:**
"At the top is a real header line from a test resume, tagged by our fine-tuned model. We use **BIO tagging**: B marks the beginning of an entity, I marks inside, and O means outside any entity. So 'Ashok' is B-NAME, 'Kunam' is I-NAME, 'Team Lead' is a DESIGNATION, and 'Microsoft' is a COMPANY.
The table lists our 10 entity types with their span counts: NAME, EMAIL, LOCATION, COLLEGE_NAME, DEGREE, GRADUATION_YEAR, COMPANIES_WORKED_AT, DESIGNATION, YEARS_OF_EXPERIENCE and SKILLS.
Why is this hard? Five reasons, listed at the bottom:
- **Domain shift.** The pretrained model learned from news (CoNLL-2003), not resumes.
- **Label mismatch.** 7 of our 10 types don't exist in CoNLL at all. CoNLL only has person, organisation, location and miscellaneous.
- **Ambiguity.** 'Visvesvaraya Technological University' is an organisation, but here it should be a COLLEGE, not a COMPANY.
- **Layout noise** from PDF text extraction.
- **Length.** The average resume has 688 words, but BERT can only read 512 sub-word tokens at a time."

### Slide 5: Real-Time Application: Resume NER API and Demo
**Thijesh:**
"This is the system architecture. A user uploads a PDF, DOCX or TXT file to our **FastAPI** endpoint, `POST /parse`. We extract the text, run the BERT model with a sliding window, add a small regex layer for emails and years, and return clean JSON with the name, email, phone, skills, education, companies, designations and every entity span.
The `model` parameter can be `pretrained`, `finetuned` or `both`, so the two models can be compared on the same resume.
On CPU it takes about **1.19 seconds per resume**. The JSON format is similar to what commercial resume parsers such as Affinda and Textkernel return, so this is a realistic application."

### Slide 6: Dataset Overview and Label Distribution
**Thijesh:**
"Our dataset is DataTurks 'Resume Entities for NER' from Kaggle. It had 220 resumes and one exact duplicate, so we used **219**. That's about **150,000 word tokens** and **3,429 entity spans**.
We split 70/15/15: 153 resumes for training, 33 for validation and 33 for test, with a fixed seed of 42. We also ran 5-fold cross-validation.
The chart shows two problems. First, **heavy imbalance**: 86% of words are 'O', meaning not part of any entity, and YEARS_OF_EXPERIENCE has only 40 spans in the whole dataset. Second, **annotation noise**: every one of the 250 EMAIL spans is actually an Indeed profile URL, not a real email, and 26 resumes leave that URL unannotated. This noise matters later when we explain the results."

### Slide 7: Preprocessing Pipeline
**Thijesh:**
"Preprocessing has six steps:
1. **De-duplicate.** Remove the one duplicate resume.
2. **Clean text.** Replace noise characters with spaces so the character offsets stay valid.
3. **Fix spans.** The dataset uses *inclusive* end offsets, has overlapping spans and has some offsets shifted by a few characters. We trimmed spans, kept the longer span when two overlapped, and expanded spans to whole-word boundaries.
4. **Tokenise** into words, keeping each word's character offsets.
5. **Convert spans to BIO**, which gives 21 labels: B and I for each of the 10 types, plus O.
6. **WordPiece alignment.** BERT splits words into sub-tokens. Only the first sub-token of each word carries the label, and the rest get −100 so the loss ignores them.
The table shows our sanity checks: 103 overlapping spans dropped, 201 spans expanded to word boundaries, a **99.1% round-trip** from BIO back to the original spans, and **zero** duplicate texts between train and test, so no data leakage.
We deliberately **did not** remove stopwords, lemmatise or lowercase. Entities contain function words, as in 'Bachelor *of* Engineering', and capital letters are a strong signal for names and companies, which is why the model is cased. For long resumes we use **sliding windows of 512 tokens with a stride of 128**."

### Slide 8: Model: BERT and the Pretrained vs Fine-Tuned Idea
**Thijesh:**
"Our base model is **`dslim/bert-base-NER`** from Hugging Face. It's BERT-base, a Transformer encoder with 12 layers, 768-dimensional hidden states, 12 attention heads and about 108 million parameters. Each layer uses self-attention: softmax of Q times K-transpose over root d-k, times V. Every word looks at every other word in the window to build its representation.
The idea is simple:
- **Before:** we use the model as-is, with its CoNLL head of 9 labels. We map PER to NAME, ORG to COMPANY and LOC to LOCATION. Everything else it simply cannot predict.
- **After:** we keep the same pretrained encoder but replace the head with a new **21-label** classification layer, then fine-tune the whole model on our resumes with cross-entropy loss and the AdamW optimiser.
Both versions are evaluated on the **same test split with the same metric**, so the comparison is fair."

### Slide 9: Implementation: Training Setup and Hyperparameters
**Thijesh:**
"On the left is our final configuration: learning rate 3e-5, batch size 8, 10 epochs, max length 512 with stride 128, weight decay 0.01, 10% warm-up, AdamW with bf16 precision. Training took only **206 seconds** on an RTX 4050 laptop GPU.
On the right is the grid search over three learning rates and two batch sizes. Note that we picked the configuration by **validation F1**, not test F1. lr 5e-5 with batch size 8 had a slightly higher *test* score, 45.9, but choosing it would mean using the test set to tune, which is cheating. So we kept 3e-5 / bs 8, which had the best validation F1 of 42.4.
The training curve shows validation F1 still climbing at epoch 10, so we kept the best checkpoint.
Now Uday will take you through the results."

---

## Part 2: Uday (Slides 10–18)

### Slide 10: Results: Before vs After Fine-Tuning
**Uday:**
"Thanks, Thijesh. This is the main result. We use **strict entity-level F1** with the `seqeval` library, so an entity counts as correct only if both the boundary and the type match exactly.
- **Micro F1** across all types went from **10.3% to 44.8%**.
- **Macro F1**, which weights every type equally, went from **5.2% to 44.5%**.
- To be fair to the pretrained model, we also scored it only on the 3 types it knows: NAME, COMPANY and LOCATION. It still only reaches **12.1%**, against **56.7%** after fine-tuning.
- Precision rose from 6.9 to 38.7, and recall from 20.2 to 53.1.
- Adding our regex rule layer for EMAIL and GRADUATION_YEAR gives a small extra boost to **45.6%**.
The chart shows F1 per entity type. The biggest jump is NAME, from about 25% to **83%**. That's 33 test resumes and 501 gold entities."

### Slide 11: Results: 5-Fold Cross-Validation and Efficiency
**Uday:**
"With only 33 test resumes, one split could be lucky, so we ran **5-fold cross-validation**. The fine-tuned model scores **40.7 ± 2.7%** micro-F1 across the folds, against **10.2 ± 1.1%** for the pretrained one. The improvement holds on every fold, as the chart shows.
For efficiency: **1.19 seconds per resume on CPU**, with a 95th percentile of 3 seconds, a **411 MB** model, and 206 seconds of training. Fine-tuning adds almost no inference cost, because only the final layer changes size."

### Slide 12: Qualitative Examples: Same Snippet, Both Models
**Uday:**
"Here are real examples from the test set. In the first snippet, the pretrained model finds the name and Microsoft, but it calls 'Kunam' a LOCATION and 'Software Engineering' a COMPANY, and it misses the designation and the Indeed URL completely. The fine-tuned model gets the name, Team Lead as DESIGNATION, Microsoft, the URL as EMAIL, and even '2.5 years' as YEARS_OF_EXPERIENCE, which isn't in the gold labels.
In the education snippet, the pretrained model calls the university and 'Engineering' a COMPANY, because news-trained NER only knows 'organisation'. The fine-tuned model gets the DEGREE, COLLEGE and GRADUATION_YEAR right.
This shows the gap is not just about missing labels. The pretrained model's *guidelines* are different."

### Slide 13: Ablations and Error Analysis
**Uday:**
"The chart shows our ablations, where we changed one thing at a time:
- **5 epochs instead of 10** drops test F1 to 31.1, so the model needs the longer training.
- **Plain truncation at 512** instead of a sliding window drops it to 38.5, because the end of long resumes gets cut off.
- **Class-weighted loss** actually hurt, at 33.9.
- **Oversampling rare-entity resumes** gave the best validation F1 but a lower test F1, so we didn't adopt it.
- **Removing text normalisation** made almost no difference.
In the error table, 266 entities are exactly correct. The biggest error types are **spurious** predictions (224), **boundary errors** (121) and **missed** entities (104). Type confusions are rare, only 10.
The hardest classes are SKILLS at 11%, YEARS_OF_EXPERIENCE at 13% and COLLEGE_NAME at 37%. SKILLS is hard because annotators sometimes labelled whole sections of up to 352 words as one skill span, so the boundaries almost never match exactly. Some 'errors' are really annotation gaps: 26 unlabelled URLs, and URLs labelled as LOCATION in 4 resumes."

### Slide 14: Live Demo: Pretrained vs Fine-Tuned Toggle
**Uday:**
"Now let me show you the working system."
→ **Switch to the browser and run the demo (see the demo script below).** If it fails, talk over the two screenshots on this slide.

### Slide 15: Discussion and Challenges
**Uday:**
"Why does fine-tuning help so much?
- **New labels.** 7 of 10 types simply don't exist in CoNLL.
- **The guidelines shift.** In news every organisation is ORG, but here only *employers* count as companies, and only the candidate's own city counts as a location.
- **The encoder's knowledge is reused.** We got a 4x gain from just 153 training resumes, because BERT already understands English.
Our main challenges were shifted and overlapping character spans, aligning sub-words to words to characters, resumes as long as 3,684 words, and a small, noisy, imbalanced dataset.
On data quality: EMAIL is only Indeed URLs, 31 resumes have a work-experience section with no company labelled, and graduation year is labelled in only 104 of the 219 resumes. The model is penalised for correct predictions the annotators missed."

### Slide 16: Conclusion and Future Work
**Uday:**
"To conclude: micro-F1 went from **10.3 to 44.8**, macro-F1 from **5.2 to 44.5**, and NAME reached **82.9%** F1.
What we learned:
- A pretrained NER model **does not transfer** when the labels and guidelines differ.
- Fine-tuning is **data-efficient**: a few hundred examples give a large gain.
- A sound method matters: strict entity F1, picking the configuration on validation only, and cross-validation.
- **Data quality is the ceiling.** Many remaining errors come from the labels themselves.
For future work: layout-aware models like **LayoutLM** that use the visual structure of the PDF, cleaner and larger annotation, LLM-based JSON extraction as a comparison, a CRF layer or partial-match metrics, and a **fairness and bias audit**, which is essential for anything used in hiring."

### Slide 17: References
**Uday:**
"These are our references: the BERT and Transformer papers, the CoNLL-2003 shared task, the dslim model card, the DataTurks dataset, Hugging Face Transformers, seqeval, LayoutLM, and work on bias in algorithmic hiring."

### Slide 18: Thank You
**Uday:**
"Thank you. We're happy to take questions."

---

## Demo script (Uday, during Slide 14, about 3–4 minutes)

Have the server already running and the tabs already open (see the checklist).

**Step 1: Side-by-side comparison on a test resume**
Open `http://127.0.0.1:8000/?sample=test_resume_082.txt&mode=compare&view=text`
> "This is a held-out test resume the model never saw in training. On the left is the pretrained model, on the right the fine-tuned one. Each colour is an entity type. On the left, it only finds a few names and organisations, and many of those are wrong. On the right, it finds names, designations, companies, colleges, degrees and years. Because this resume is from the test set, the page also shows the **gold annotation** and the **per-document F1** for each model."

**Step 2: Structured fields view**
Switch the view toggle to **Fields** (or use `&view=fields` in the URL).
> "This is what an ATS would actually use: name, email, phone, skills, education, companies and designations, pulled into clean fields."

**Step 3: Upload a file**
Click upload and pick `data\sample_resumes\sample_fictional.pdf` (or the `.docx`), with mode **Fine-tuned**.
> "It works on real PDF and Word files too. This is a fictional resume we made, so it contains no real personal data. Parsing takes about a second on CPU."

**Step 4: Results tab**
Open `http://127.0.0.1:8000/#results`.
> "The Results tab shows the same before-vs-after tables, cross-validation, ablations and figures from our slides, generated straight from the result files, so none of the numbers were typed by hand."

**Step 5 (optional): the API**
Open `http://127.0.0.1:8000/docs`, or run in a terminal:
```powershell
curl.exe -F file=@data/sample_resumes/sample_fictional.pdf -F model=finetuned http://127.0.0.1:8000/parse
```
> "Everything the web page does goes through this REST API, so another system could call it directly. `model=both` returns both models' outputs in one response."

Then go back to the slides: "Let me go back to the slides for the discussion."

**If something breaks:** stay calm, go back to Slide 14, and say *"Here are screenshots of the same demo."* Then describe the comparison and fields views using the screenshots.

---

## Likely questions (either speaker)

| Question | Short answer |
|---|---|
| Why is 44.8% F1 "good"? | It's strict exact-match F1 on noisy labels. The point is the 4x relative gain over the pretrained model, which holds in all 5 CV folds. NAME reaches 83%. |
| Why not pick lr 5e-5, which had 45.9 test F1? | Choosing by test score means tuning on the test set. We chose only by validation F1. |
| Why `dslim/bert-base-NER` and not plain BERT? | It already knows NER (CoNLL), so it gives a meaningful "before" baseline, and the comparison isolates the effect of fine-tuning. |
| How do you handle resumes longer than 512 tokens? | Sliding windows of 512 with a stride of 128, then merging the predictions back at word level. Truncation instead costs about 6 F1 points (ablation). |
| Why no lowercasing or stopword removal? | Capitalisation is a key cue for names and companies (the model is cased), and entities contain stopwords ("Bachelor of Engineering"). |
| Why is SKILLS so low? | Annotators labelled whole skill sections (up to 352 words) as single spans, so exact boundaries rarely match. |
| What does the regex layer do? | It catches EMAIL/URL patterns, graduation years and phone numbers, which are pattern-shaped. It adds about +0.8 F1. |
| Where would you go next? | LayoutLM for PDF layout, cleaner annotation, an LLM-based extraction baseline, and a bias audit before any hiring use. |
