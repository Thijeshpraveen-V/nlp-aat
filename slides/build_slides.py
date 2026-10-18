"""Build slides/AAT_presentation.pptx from the college template (LLM AAT_template.pptx).

Keeps the template's title slide, Agenda slide and closing slide (in that order), fills in the
team details, and inserts content slides styled like the template (Times New Roman, dark-red
centred titles). All numbers are read from results/*.json. Charts are native PowerPoint charts.

    python slides/build_slides.py
"""
import copy
import json
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
R = lambda p: json.loads((RES / p).read_text(encoding="utf-8"))
team = json.loads((ROOT / "report" / "team.json").read_text(encoding="utf-8"))
S, Q, P = R("data_stats.json"), R("data_quality.json"), R("metrics_pretrained.json")
FC = R("final_config.json"); M = R(f"runs/{FC['final_run']}.json")
BA, AB, EF, EA = R("metrics_before_after.json"), R("metrics_ablations.json"), R("metrics_efficiency.json"), R("error_analysis.json")
CVR = R("runs/cv_main.json"); QE = R("qualitative_examples.json")
before, after, afterR = (BA["views"]["all_types"][k] for k in ("before", "after", "after_rules"))
ov = BA["views"]["overlap_types"]
pct = lambda v: f"{100 * v:.1f}"
TYPES = ["NAME", "EMAIL", "LOCATION", "COLLEGE_NAME", "DEGREE", "GRADUATION_YEAR",
         "COMPANIES_WORKED_AT", "DESIGNATION", "YEARS_OF_EXPERIENCE", "SKILLS"]
SHORT = {"COMPANIES_WORKED_AT": "COMPANY", "YEARS_OF_EXPERIENCE": "YRS EXP.", "GRADUATION_YEAR": "GRAD. YEAR",
         "COLLEGE_NAME": "COLLEGE"}

RED, BLACK, GREY, DARK = RGBColor(0xC0, 0, 0), RGBColor(0, 0, 0), RGBColor(0x9A, 0xA5, 0xB1), RGBColor(0x40, 0x40, 0x40)
TINT, BLUE = RGBColor(0xFB, 0xEC, 0xEC), RGBColor(0x1F, 0x4E, 0x79)
FONT = "Times New Roman"

prs = Presentation(ROOT / "LLM AAT_template.pptx")
BLANK = next(l for l in prs.slide_layouts if l.name == "BLANK")


# ------------------------------------------------------------------ helpers
def style(run, size, bold=False, color=BLACK, italic=False):
    f = run.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = color
    return run


def title(slide, text, size=30):
    tb = slide.shapes.add_textbox(Inches(0), Inches(0.05), Inches(10), Inches(0.62))
    tb.name = "Title"
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    style(p.add_run(), size, color=RED).text = text
    p.runs[0].text = text
    return tb


def _bullet(p, char="•"):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", "228600"); pPr.set("indent", "-228600")
    for tag in ("a:buNone", "a:buChar", "a:buFont"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    bf = etree.SubElement(pPr, qn("a:buFont")); bf.set("typeface", "Arial")
    bc = etree.SubElement(pPr, qn("a:buChar")); bc.set("char", char)


def text(slide, x, y, w, h, items, size=16, bullets=True, space=6, bold_first=False, color=BLACK, name="Body"):
    """items: list of str or list of (text, bold) runs. '**x**' marks bold."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        if bullets:
            _bullet(p)
        parts = item.split("**")
        for k, part in enumerate(parts):
            if part:
                style(p.add_run(), size, bold=(k % 2 == 1), color=color).text = part
                p.runs[-1].text = part
    return tb


def picture(slide, path, x, y, w=None, h=None, name=None):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
    pic.name = name or Path(path).stem
    return pic


def caption(slide, x, y, w, s):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.3))
    p = tb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    style(p.add_run(), 11, italic=True, color=DARK).text = s
    p.runs[0].text = s
    tb.text_frame.word_wrap = True


def stat(slide, x, y, w, big, label, color=RED, h=1.15):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = f"Stat {label[:20]}"
    box.fill.solid(); box.fill.fore_color.rgb = TINT
    box.line.fill.background()
    box.adjustments[0] = 0.12
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    style(p.add_run(), 30, bold=True, color=color).text = big
    p.runs[0].text = big
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    style(p2.add_run(), 12, color=DARK).text = label
    p2.runs[0].text = label


def table(slide, x, y, w, rows, col_w, size=12, header_fill=RED, highlight_rows=(), name="Table", row_h=0.28, header_h=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    shape.name = name
    t = shape.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(header_h if (i == 0 and header_h) else row_h)
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = header_fill if i == 0 else (TINT if i in highlight_rows else RGBColor(0xFF, 0xFF, 0xFF))
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            run = p.add_run(); run.text = str(v)
            style(run, size, bold=(i == 0 or i in highlight_rows), color=RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else BLACK)
    return shape


def bar_chart(slide, x, y, w, h, cats, series, colors, horizontal=False, maxv=100, legend=True, label_fmt='0.0', name="Chart", font=11):
    cd = CategoryChartData(); cd.categories = cats
    for nm, vals in series:
        cd.add_series(nm, vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = name
    ch = gf.chart
    ch.font.name = FONT; ch.font.size = Pt(font)
    ch.has_legend = legend and len(series) > 1
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(font)
    va = ch.value_axis
    va.maximum_scale = maxv; va.minimum_scale = 0
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xD9, 0xD9, 0xD9)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(font - 1)
    ch.category_axis.tick_labels.font.size = Pt(font - 1)
    ch.category_axis.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    plot = ch.plots[0]
    plot.gap_width = 60; plot.overlap = -10 if len(series) > 1 else 0
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = label_fmt; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(font - 2)
    for s, c in zip(plot.series, colors):
        s.format.fill.solid(); s.format.fill.fore_color.rgb = c
    return gf


def new_slide(t, notes, size=30):
    s = prs.slides.add_slide(BLANK)
    title(s, t, size)
    s.notes_slide.notes_text_frame.text = notes
    return s


def set_runs(shape, mapping):
    for p in shape.text_frame.paragraphs:
        for r in p.runs:
            for old, new in mapping.items():
                if r.text.strip() == old:
                    r.text = new


# ------------------------------------------------------------------ slide 1 (template): title
s1 = prs.slides[0]
m1, m2 = team["members"]
for sh in s1.shapes:
    if not sh.has_text_frame:
        continue
    txt = sh.text_frame.text
    if txt.startswith("TOPIC"):
        p0 = sh.text_frame.paragraphs[0]
        p0.runs[0].text = "Resume NER: Pretrained vs Fine-Tuned BERT"
        for r in p0.runs[1:]:
            r.text = ""
        p0.runs[0].font.size = Pt(26)
    if txt.startswith("Presented By"):
        seen = 0
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if "Student Name & USN" in r.text:
                    m = (m1, m2)[seen]; seen += 1
                    r.text = f"{m['name']} – {m['usn']}"
s1.notes_slide.notes_text_frame.text = (
    f"Title: {team['topic']}. Presented by {m1['name']} ({m1['usn']}) and {m2['name']} ({m2['usn']}), "
    "7th semester, section C. Course: Natural Language Processing for LLM (24AM7PCLLM).")

# slide 2 (template): Agenda — kept unchanged; slide 3 (template): Thank you — moved to the end later
prs.slides[1].notes_slide.notes_text_frame.text = (
    "The talk follows the structure of the report: introduction, the NER task, our application, the BERT model, "
    "data and preprocessing, implementation, results, discussion, conclusion and references.")

# ------------------------------------------------------------------ content slides
# 1 Introduction / problem
s = new_slide("Introduction: Why Resume Parsing Needs NER",
              "Recruiters and applicant tracking systems receive resumes as unstructured PDFs and Word files. "
              "To search or match candidates, the text must become structured fields. NER does that extraction. "
              "Our question: how well does an off-the-shelf NER model do on resumes, and how much does fine-tuning help?")
text(s, 0.5, 0.95, 5.2, 3.9, [
    "**Problem:** resumes arrive as unstructured PDF / DOCX text with no fixed layout",
    "**ATS need:** structured fields — name, contact, education, employers, titles, skills",
    "**NER** turns raw text into typed entities with character spans",
    "**Question:** how good is a pretrained NER model on resumes, and how much does fine-tuning add?",
    "**Deliverables:** measured before-vs-after comparison + REST API + live web demo",
], size=16, space=10)
stat(s, 6.1, 1.0, 3.4, f"{S['n_resumes']}", "annotated resumes (DataTurks, Kaggle)")
stat(s, 6.1, 2.35, 3.4, f"{len(TYPES)}", "resume entity types, BIO tagging")
stat(s, 6.1, 3.7, 3.4, f"{pct(before['micro']['f1'])} → {pct(after['micro']['f1'])}", "test micro-F1 (%), before → after fine-tuning")

# 2 NER task + entity types
s = new_slide("Selected NLP Topic: Named Entity Recognition",
              "NER is sequence labelling. Every word gets a BIO label: B for the first word of an entity, I for the "
              "following words, O for everything else. Ten entity types give 21 labels. The example shows a real "
              "header from the test set, tagged by the fine-tuned model.")
ex = next(q for q in QE if q["doc"] == "resume_009")
words = [("Ashok", "B-NAME"), ("Kunam", "I-NAME"), ("Team", "B-DESIGNATION"), ("Lead", "I-DESIGNATION"), ("-", "O"), ("Microsoft", "B-COMPANIES_WORKED_AT")]
COL = {"NAME": RGBColor(0xFF, 0xD8, 0xA8), "DESIGNATION": RGBColor(0xC0, 0xEB, 0x75), "COMPANIES_WORKED_AT": RGBColor(0x99, 0xE9, 0xF2), "O": RGBColor(0xEE, 0xEE, 0xEE)}
x = 0.5
for wd, tag in words:
    typ = tag[2:] if tag != "O" else "O"
    wbox = 0.35 + 0.13 * max(len(wd), len(SHORT.get(typ, typ)) * 0.75)
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(0.95), Inches(wbox), Inches(0.42))
    b.name = f"Word {wd}"; b.fill.solid(); b.fill.fore_color.rgb = COL[typ]; b.line.fill.background()
    p = b.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    style(p.add_run(), 15, bold=True).text = wd; p.runs[0].text = wd
    lab = s.shapes.add_textbox(Inches(x - 0.1), Inches(1.4), Inches(wbox + 0.2), Inches(0.3))
    lp = lab.text_frame.paragraphs[0]; lp.alignment = PP_ALIGN.CENTER
    lt = tag if tag == "O" else tag[:2] + SHORT.get(typ, typ)
    style(lp.add_run(), 10, color=DARK).text = lt; lp.runs[0].text = lt
    x += wbox + 0.12
text(s, x + 0.2, 0.95, 9.6 - x, 0.8, ["Real header line from a test resume, tagged by the fine-tuned model"], size=12, bullets=False, color=DARK)
rows = [["Entity type", "Spans", "Example"]] + [[t, S["entities_per_type"][t], e] for t, e in [
    ("NAME", "Ashok Kunam"), ("EMAIL", "indeed.com/r/…"), ("LOCATION", "Bengaluru"), ("COLLEGE_NAME", "Visvesvaraya Technological Univ."),
    ("DEGREE", "Bachelor of Engineering"), ("GRADUATION_YEAR", "2017"), ("COMPANIES_WORKED_AT", "Microsoft"),
    ("DESIGNATION", "Server Support Engineer"), ("YEARS_OF_EXPERIENCE", "2.5 years"), ("SKILLS", "Python, SQL, …")]]
table(s, 0.5, 1.9, 6.0, rows, [2.3, 0.8, 2.9], size=11, row_h=0.29)
text(s, 6.8, 1.95, 2.9, 3.2, [
    "**Domain shift:** news (CoNLL-2003) → resumes",
    "**Label mismatch:** 7 of 10 types missing in CoNLL",
    "**Ambiguity:** company vs college vs skill",
    "**Layout noise** from PDF extraction",
    f"**Long inputs:** mean {round(S['tokens_per_resume']['mean'])} words (BERT limit 512 sub-tokens)",
], size=13, space=6)

# 3 Application / API
s = new_slide("Real-Time Application: Resume NER API and Demo",
              "Our application: FastAPI backend with POST /parse, React front end. The same pipeline serves the CLI, "
              "API and demo. Both models are available so the demo can show them side by side. Commercial parsers "
              "such as Affinda and Textkernel offer the same idea as paid REST APIs returning JSON.")
picture(s, FIG / "system_architecture.png", 0.4, 0.8, w=9.2)
text(s, 0.5, 4.15, 4.6, 1.3, [
    "**POST /parse** — PDF / DOCX / TXT upload → JSON",
    "**model =** pretrained | finetuned | both",
    f"**CPU latency:** {EF['finetuned']['sec_per_resume_mean']:.2f} s per resume (mean)",
], size=13, space=3)
text(s, 5.2, 4.15, 4.4, 1.3, [
    "Output: name, email, phone, skills, education, companies, designations + entity spans",
    "Context: commercial resume-parsing APIs (Affinda, Textkernel) return similar JSON",
], size=13, space=3)

# 4 Dataset
s = new_slide("Dataset Overview and Label Distribution",
              f"DataTurks Resume Entities for NER from Kaggle: {S['n_resumes_raw']} resumes, one duplicate removed. "
              f"86 percent of words are outside any entity. Companies are frequent, years of experience very rare, "
              "and skills spans are very long because whole sections were annotated.")
picture(s, FIG / "label_distribution.png", 0.3, 0.8, w=6.4)
caption(s, 0.3, 3.15, 6.4, "Entity spans per type (left) and word tokens per type, log scale (right)")
stat(s, 7.0, 0.85, 2.6, f"{S['n_resumes']}", f"resumes ({S['n_resumes_raw']} raw, {S['n_duplicates_removed']} duplicate)", h=1.0)
stat(s, 7.0, 2.0, 2.6, f"{S['n_word_tokens']:,}", "word tokens", h=1.0)
stat(s, 7.0, 3.15, 2.6, f"{S['entities_total']:,}", "entity spans", h=1.0)
text(s, 0.5, 3.55, 6.2, 1.8, [
    f"**Split:** {S['split_sizes']['train']} / {S['split_sizes']['val']} / {S['split_sizes']['test']} resumes (70/15/15, seed 42) + 5-fold CV",
    f"**Imbalance:** {pct(S['fraction_O_tokens'])} % of words are O; YEARS_OF_EXPERIENCE only {S['entities_per_type']['YEARS_OF_EXPERIENCE']} spans",
    f"**Noise:** all {Q['email_spans']} EMAIL spans are Indeed URLs; {Q['resumes_with_url_not_annotated']} resumes leave the URL unannotated",
], size=13, space=4)

# 5 Preprocessing
s = new_slide("Preprocessing Pipeline",
              "Character spans are converted to word-level BIO labels. Cleaning replaces noise characters one for one "
              "so offsets stay valid. Stopwords are NOT removed and no lemmatisation is done, because NER must label "
              "every word and entities contain function words. A round trip back to spans recovers 99 percent.")
steps = [("1", "De-duplicate", f"{S['n_duplicates_removed']} duplicate removed"),
         ("2", "Clean text", "noise chars → space, offsets kept"),
         ("3", "Fix spans", "inclusive end, trim, resolve overlaps"),
         ("4", "Tokenise", "regex words + char offsets"),
         ("5", "Span → BIO", "21 labels, word-aligned"),
         ("6", "WordPiece", "first sub-token labelled, rest −100")]
for i, (n, h, d) in enumerate(steps):
    xx = 0.35 + i * 1.57
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(xx + 0.5), Inches(0.95), Inches(0.5), Inches(0.5))
    c.name = f"Step {n}"; c.fill.solid(); c.fill.fore_color.rgb = RED; c.line.fill.background()
    p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    style(p.add_run(), 16, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF)).text = n; p.runs[0].text = n
    text(s, xx, 1.5, 1.5, 0.35, [h], size=14, bullets=False).text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    tb = text(s, xx, 1.85, 1.5, 0.8, [d], size=11, bullets=False, color=DARK)
    tb.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    if i < len(steps) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(xx + 1.17), Inches(1.1), Inches(0.3), Inches(0.2))
        a.fill.solid(); a.fill.fore_color.rgb = GREY; a.line.fill.background()
al = S["alignment"]
rows = [["Alignment check (data_prep.py log)", "Count"],
        ["Overlapping spans dropped", al["dropped_overlap"]],
        ["Spans expanded to word boundaries", al["span_misaligned_expanded_to_word_boundary"]],
        ["BIO → span round trip recovered", f"{al['roundtrip_recovered_pct']:.1f} %"],
        ["Train/test identical texts (leakage)", S["train_test_identical_texts"]]]
table(s, 0.5, 2.85, 5.0, rows, [3.7, 1.3], size=12)
text(s, 5.8, 2.85, 3.9, 2.4, [
    "**No stopword removal / lemmatisation:** entities contain function words (\"Bachelor *of* Engineering\")",
    "**No lower-casing:** capitalisation is a key cue; the model is cased",
    "**Long resumes:** sliding windows of 512 sub-tokens, stride 128",
], size=13, space=6)

# 6 BERT
s = new_slide("Model: BERT and the Pretrained vs Fine-Tuned Idea",
              "dslim/bert-base-NER is BERT-base-cased fine-tuned on CoNLL-2003 news NER. Before: we use it as is and "
              "map PER, ORG, LOC to our types. After: we keep the encoder, replace the 9-label head by a 21-label "
              "head and fine-tune the whole network on resumes. Self-attention lets each token use context from both sides.")
picture(s, FIG / "bert_ner_architecture.png", 0.3, 0.85, w=6.5)
text(s, 7.0, 0.85, 2.75, 4.6, [
    "**12 layers**, 768-d, 12 heads, 108 M params",
    "**Before:** CoNLL head (9 labels), PER→NAME, ORG→COMPANY, LOC→LOCATION",
    f"**After:** new head (21 labels), whole model fine-tuned, cross-entropy + AdamW",
    "Attention = softmax(QKᵀ/√dₖ)V",
], size=13, space=8)

# 7 Training setup
s = new_slide("Implementation: Training Setup and Hyperparameters",
              "Learning rate and batch size were chosen by a grid search on the validation set only. The best was "
              f"learning rate {M['config']['lr']:.0e}, batch size {M['config']['batch_size']}, 10 epochs. Training took about "
              f"{round(M['train_time_sec'] / 60)} minutes on the laptop GPU. Validation F1 kept rising to the last epoch while validation loss "
              "rose after epoch 4, so we select checkpoints by F1.")
c = M["config"]
rows = [["Hyperparameter", "Value"], ["Learning rate", f"{c['lr']:.0e}"], ["Batch size", c["batch_size"]],
        ["Epochs (best kept)", f"{c['epochs']} (best {M['best_epoch']})"], ["Max length / stride", f"{c['max_length']} / {c['stride']}"],
        ["Weight decay / warm-up", f"{c['weight_decay']} / {int(100 * c['warmup'])} %"], ["Optimiser", "AdamW, bf16"],
        ["Train time (RTX 4050)", f"{round(M['train_time_sec'])} s"], ["Selection", "best validation F1"]]
table(s, 0.4, 0.85, 3.5, rows, [1.95, 1.55], size=11, row_h=0.3)
picture(s, FIG / "training_curves.png", 4.1, 0.85, w=5.6)
grid = FC["grid"]
rows = [["lr", "bs", "Val F1", "Test F1"]] + [[f"{g['lr']:.0e}", g["batch_size"], pct(g["best_val_micro_f1"]), pct(g["test_micro_f1"])] for g in grid]
best_i = 1 + next(i for i, g in enumerate(grid) if g["tag"] == FC["best"]["tag"])
table(s, 4.6, 3.05, 4.6, rows, [1.1, 0.8, 1.35, 1.35], size=10, row_h=0.31, highlight_rows=(best_i,))

# 8 Before vs after per type (native chart)
s = new_slide("Results: Before vs After Fine-Tuning",
              f"Same test split, same metric: seqeval strict entity-level F1. The pretrained model only scores on NAME, "
              f"LOCATION and COMPANY, and is zero on the seven types it cannot express. After fine-tuning every type improves; "
              f"NAME reaches {pct(after['per_type']['NAME']['f1'])} percent. SKILLS and years of experience stay hard.")
bar_chart(s, 0.3, 0.75, 6.3, 4.7, [SHORT.get(t, t) for t in TYPES],
          [("Before: pretrained", [round(100 * before["per_type"][t]["f1"], 1) for t in TYPES]),
           ("After: fine-tuned", [round(100 * after["per_type"][t]["f1"], 1) for t in TYPES])],
          [GREY, RED], name="F1 per type", font=10)
rows = [["Metric (%)", "Before", "After"],
        ["Micro F1, all types", pct(before["micro"]["f1"]), pct(after["micro"]["f1"])],
        ["Macro F1, all types", pct(before["macro"]["f1"]), pct(after["macro"]["f1"])],
        ["Micro F1, 3 shared types", pct(ov["before"]["micro"]["f1"]), pct(ov["after"]["micro"]["f1"])],
        ["Precision (micro)", pct(before["micro"]["precision"]), pct(after["micro"]["precision"])],
        ["Recall (micro)", pct(before["micro"]["recall"]), pct(after["micro"]["recall"])],
        ["+ regex rules (micro F1)", "—", pct(afterR["micro"]["f1"])]]
table(s, 6.75, 0.9, 3.0, rows, [1.7, 0.65, 0.65], size=11, row_h=0.36, highlight_rows=(1,))
caption(s, 6.75, 3.55, 3.0, f"Test split: {S['split_sizes']['test']} resumes, {after['micro']['support']} gold entities")

# 9 Cross-validation
pcv, fcv = P["cv_summary"]["all_types"], CVR["summary"]["test_micro_f1"]
s = new_slide("Results: 5-Fold Cross-Validation and Efficiency",
              f"Cross-validation gives a more reliable estimate: {pct(fcv['mean'])} plus or minus {pct(fcv['std'])} for the "
              f"fine-tuned model against {pct(pcv['micro_f1_mean'])} for the pretrained model; fine-tuning wins on every fold. "
              "Inference cost is the same for both models because the architecture is the same.")
folds = [f"Fold {i + 1}" for i in range(5)]
bar_chart(s, 0.3, 0.75, 5.6, 4.7, folds,
          [("Before: pretrained", [round(100 * f["all_types"]["micro"]["f1"], 1) for f in P["cv_folds"]]),
           ("After: fine-tuned", [round(100 * v, 1) for v in fcv["folds"]])], [GREY, RED], name="CV folds")
stat(s, 6.2, 0.85, 3.5, f"{pct(fcv['mean'])} ± {pct(fcv['std'])}", f"fine-tuned CV micro-F1 (%) vs {pct(pcv['micro_f1_mean'])} ± {pct(pcv['micro_f1_std'])} before", h=1.2)
stat(s, 6.2, 2.2, 3.5, f"{EF['finetuned']['sec_per_resume_mean']:.2f} s", f"CPU per resume (p95 {EF['finetuned']['sec_per_resume_p95']:.1f} s)", h=1.05)
stat(s, 6.2, 3.4, 3.5, f"{round(EF['finetuned']['model_size_mb'])} MB", f"model size · trained in {round(M['train_time_sec'])} s", h=1.05)

# 10 Qualitative
s = new_slide("Qualitative Examples: Same Snippet, Both Models",
              "The pretrained model tags universities as companies and splits header lines. The fine-tuned model "
              "reproduces the resume labels: degrees, colleges, graduation years and Indeed profile links.")
fmt = lambda es: "\n".join(f"{e['text'].replace(chr(10), ' ')[:34]} → {SHORT.get(e['label'], e['label'])}" for e in es[:6]) or "—"
rows = [["Snippet", "Gold", "Pretrained (before)", "Fine-tuned (after)"]]
for doc in ("resume_009", "resume_055"):
    q = next(q for q in QE if q["doc"] == doc)
    snip = "\n".join(l for l in q["snippet"].strip().split("\n") if l.strip())[:170]
    rows.append([snip, fmt(q["gold"]), fmt(q["pretrained"]), fmt(q["finetuned"])])
table(s, 0.3, 0.8, 9.4, rows, [2.35, 2.35, 2.35, 2.35], size=10, row_h=2.0, header_h=0.35, name="Qualitative")

# 11 Ablations + errors
s = new_slide("Ablations and Error Analysis",
              "Each ablation changes one setting. Sliding windows and enough epochs matter most; class weighting hurts; "
              "the regex rule layer adds a little through EMAIL. Most remaining errors are boundary errors and "
              "spurious predictions, many of which are real but unannotated entities.")
names = {"Final": "Final model", "+ regex": "+ regex rules", "Epochs": "5 epochs", "+ class": "class-weighted loss",
         "+ over": "oversampling", "Plain": "truncation (no sliding)", "No text": "no normalisation"}
lab = lambda n: next(v for k, v in names.items() if n.startswith(k))
abl = AB["ablations"]
bar_chart(s, 0.3, 0.75, 5.2, 4.7, [lab(a["name"]) for a in abl][::-1], [("Test micro-F1", [round(100 * a["micro_f1"], 1) for a in abl][::-1])],
          [RED], horizontal=True, maxv=60, name="Ablations", font=10)
ec = EA["categories"]
rows = [["Error category (test)", "Count"], ["Correct", ec["correct"]], ["Boundary error", ec["boundary"]],
        ["Type error (incl. boundary+type)", ec.get("type", 0) + ec.get("bound+type", 0)], ["Missed", ec["missed"]], ["Spurious", ec["spurious"]]]
table(s, 5.8, 0.9, 3.9, rows, [2.9, 1.0], size=11)
hard = EA["hardest_types_by_f1"][:3]
text(s, 5.8, 2.85, 3.9, 2.5, [
    "**Hardest:** " + ", ".join(f"{h['type']} ({pct(h['f1'])})" for h in hard),
    f"**SKILLS:** whole sections up to {Q['span_length_words']['SKILLS']['max']} words → boundary errors",
    f"**Annotation gaps:** {Q['resumes_with_url_not_annotated']} unlabelled Indeed URLs; URL labelled LOCATION in {Q['indeed_url_spans_by_gold_type'].get('LOCATION', 0)} resumes",
], size=12, space=5)

# 12 Demo
s = new_slide("Live Demo: Pretrained vs Fine-Tuned Toggle",
              "Live demo: run uvicorn app.api:app and open localhost:8000. Pick test_resume_082 and the side-by-side view: "
              "gold, pretrained and fine-tuned columns with per-document F1. Then upload the fictional PDF and switch to "
              "structured fields. Deep link: /?sample=test_resume_082.txt&mode=compare")
picture(s, RES / "screenshots" / "demo_compare.png", 0.3, 0.75, h=4.4)
picture(s, RES / "screenshots" / "demo_fields.png", 5.25, 0.75, w=4.45)
text(s, 5.25, 4.3, 4.45, 1.2, [
    "Upload PDF / DOCX / TXT or pick a sample",
    "Pretrained · fine-tuned · side-by-side toggle",
    "Gold labels + per-document F1 for test resumes",
    "Results tab: before-vs-after tables and figures",
], size=13, space=3)

# 13 Discussion / challenges
s = new_slide("Discussion and Challenges",
              "Fine-tuning helps for two reasons: the label set changes and the annotation guidelines differ from news. "
              "The challenges were span alignment, sub-word alignment, long resumes and the tiny, noisy dataset.")
cards = [("Why fine-tuning helps", ["New labels: 7 of 10 types do not exist in CoNLL", "Guidelines shift: only employers, the candidate's city",
                                    "Encoder knowledge is reused: big gain from 153 resumes"]),
         ("Challenges", ["Shifted, overlapping character spans", "Sub-word ↔ word ↔ character alignment",
                         f"Resumes up to {S['tokens_per_resume']['max']:,} words", "Small, noisy, imbalanced data"]),
         ("Data quality", [f"EMAIL = Indeed URLs only ({Q['email_spans']} spans)",
                           f"{Q['section_heading_but_no_annotation']['COMPANIES_WORKED_AT']['of_which_no_gold_span']} resumes with WORK EXPERIENCE but no company labelled",
                           f"GRADUATION_YEAR labelled in only {Q['resumes_with_at_least_one_span_of_type']['GRADUATION_YEAR']} resumes"])]
for i, (h, items) in enumerate(cards):
    xx = 0.35 + i * 3.15
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(xx), Inches(0.9), Inches(2.95), Inches(3.3))
    b.name = f"Card {h}"; b.fill.solid(); b.fill.fore_color.rgb = TINT; b.line.fill.background(); b.adjustments[0] = 0.05
    text(s, xx + 0.1, 1.0, 2.75, 0.45, [h], size=17, bullets=False, color=RED).text_frame.paragraphs[0].runs[0].font.bold = True
    text(s, xx + 0.1, 1.5, 2.75, 3.7, items, size=13, space=8)

# 14 Conclusion
s = new_slide("Conclusion and Future Work",
              "Summary: same model, same data, same metric. Fine-tuning raised micro F1 from about 10 to about 45 percent "
              "and the system runs on a laptop CPU. Next steps: layout-aware models such as LayoutLM, cleaner and larger "
              "data, LLM-based extraction for comparison, and a fairness audit before any use in hiring.")
stat(s, 0.5, 0.9, 2.9, f"{pct(before['micro']['f1'])} → {pct(after['micro']['f1'])}", "test micro-F1 (%)", h=1.1)
stat(s, 0.5, 2.15, 2.9, f"{pct(before['macro']['f1'])} → {pct(after['macro']['f1'])}", "test macro-F1 (%)", h=1.1)
stat(s, 0.5, 3.4, 2.9, f"{pct(after['per_type']['NAME']['f1'])} %", "NAME F1 after fine-tuning", h=1.1)
text(s, 3.7, 0.9, 2.9, 4.4, [
    "**Learned**",
    "Pretrained NER does not transfer when labels and guidelines differ",
    "Fine-tuning is data-efficient",
    "Strict entity F1 + validation-only selection + CV",
    "Data quality is the ceiling",
], size=14, space=6)
text(s, 6.8, 0.9, 2.9, 4.4, [
    "**Future work**",
    "Layout-aware models (LayoutLM)",
    "Cleaner, larger annotation",
    "LLM-based JSON extraction",
    "CRF / partial-match metrics",
    "Fairness and bias audit",
], size=14, space=6)

# 15 References
s = new_slide("References",
              "All references are real and verified; the full list in IEEE style is in the report.")
refs = [
    "[1] J. Devlin et al., \"BERT: Pre-training of deep bidirectional transformers for language understanding,\" NAACL-HLT, 2019.",
    "[2] A. Vaswani et al., \"Attention is all you need,\" NeurIPS, 2017.",
    "[3] E. F. Tjong Kim Sang and F. De Meulder, \"Introduction to the CoNLL-2003 shared task,\" CoNLL, 2003.",
    "[4] dslim, \"bert-base-NER,\" Hugging Face model card, huggingface.co/dslim/bert-base-NER.",
    "[5] DataTurks, \"Resume Entities for NER,\" Kaggle, kaggle.com/datasets/dataturks/resume-entities-for-ner.",
    "[6] T. Wolf et al., \"Transformers: State-of-the-art natural language processing,\" EMNLP Demos, 2020.",
    "[7] H. Nakayama, \"seqeval: A Python framework for sequence labeling evaluation,\" 2018.",
    "[8] Y. Xu et al., \"LayoutLM: Pre-training of text and layout for document image understanding,\" KDD, 2020.",
    "[9] M. Raghavan et al., \"Mitigating bias in algorithmic hiring: Evaluating claims and practices,\" FAT*, 2020.",
    "[10] Affinda and Textkernel resume-parser API documentation (docs.affinda.com, developer.textkernel.com).",
]
text(s, 0.5, 0.85, 9.0, 4.6, refs, size=12, bullets=False, space=5)

# ------------------------------------------------------------------ move "Thank you" (template slide 3) to the end
sld = prs.slides._sldIdLst
ids = list(sld)
thank = ids[2]
sld.remove(thank)
sld.append(thank)

out = ROOT / "slides" / "AAT_presentation.pptx"
prs.save(out)
print("wrote", out, "slides:", len(prs.slides))
