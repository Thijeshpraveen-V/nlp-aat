"""Diagrams for the report and slides (drawn with matplotlib):
  system_architecture.png   end-to-end resume parsing pipeline (API + demo)
  bert_ner_architecture.png BERT encoder + token-classification head (pretrained vs fine-tuned head)
  experiment_workflow.png   experimental design: before / after / ablations / CV
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIGURES

BLUE, GREY, ORANGE, GREEN, PURPLE = "#dbe4ff", "#e9ecef", "#ffe8cc", "#d3f9d8", "#e5dbff"
EDGE = "#343a40"


def box(ax, x, y, w, h, text, color, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=EDGE, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, lw=1.2, color=EDGE))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, text, ha="center", va="bottom", fontsize=7.5, style="italic")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def system_architecture():
    fig, ax = canvas(13, 4.6)
    top = [("Resume file\nPDF / DOCX / TXT", ORANGE), ("Text extraction\npdfplumber,\npython-docx", GREY),
           ("Offset-preserving\ncleaning", GREY), ("Word tokenisation\nregex \\w+|[^\\w\\s]\n+ char offsets", GREY),
           ("WordPiece +\nsliding windows\n512 tokens, stride 128", BLUE)]
    bot = [("BERT-base encoder\n12 layers, 768-d\n(dslim/bert-base-NER)", BLUE), ("Token classification\nhead (Linear+softmax)", BLUE),
           ("Window merge\navg. probabilities,\nfirst sub-token", BLUE), ("BIO repair +\nregex rules\n(EMAIL, year, phone)", GREEN),
           ("Spans → JSON\nname, email, skills,\neducation, companies", GREEN)]
    w, h, gap = 2.2, 1.25, 0.35
    for i, (t, c) in enumerate(top):
        x = 0.3 + i * (w + gap)
        box(ax, x, 3.0, w, h, t, c)
        if i:
            arrow(ax, x - gap, 3.0 + h / 2, x, 3.0 + h / 2)
    xr = 0.3 + 4 * (w + gap) + w / 2
    arrow(ax, xr, 3.0, xr, 2.35)
    for i, (t, c) in enumerate(bot):
        x = 0.3 + (4 - i) * (w + gap)
        box(ax, x, 1.1, w, h, t, c)
        if i:
            arrow(ax, x + w + gap, 1.1 + h / 2, x + w, 1.1 + h / 2)
    ax.text(6.5, 0.45, "FastAPI  POST /parse  ·  POST /parse_text  ·  React UI with pretrained / fine-tuned / side-by-side toggle",
            ha="center", fontsize=9.5, weight="bold", color="#364fc7")
    ax.text(6.5, 4.45, "Resume parsing pipeline (shared by CLI, API and web demo)", ha="center", fontsize=11, weight="bold")
    plt.savefig(FIGURES / "system_architecture.png", dpi=170, bbox_inches="tight"); plt.close()


def bert_architecture():
    fig, ax = canvas(12, 7.2)
    toks = ["[CLS]", "Ana", "##nya", "Rao", "Data", "Scientist", "-", "Fin", "##tri", "##x", "[SEP]"]  # real WordPiece output
    n = len(toks)
    x0, dx = 0.7, 0.98
    for i, t in enumerate(toks):
        box(ax, x0 + i * dx, 0.25, 0.86, 0.5, t, GREY, fs=8.5)
    box(ax, x0, 1.05, n * dx - 0.12, 0.6, "Embeddings = token (WordPiece, 28,996 cased vocab) + position + segment   →  768-d", ORANGE)
    box(ax, x0, 2.0, n * dx - 0.12, 2.0, "", BLUE)
    ax.text(x0 + 0.2, 3.75, "Transformer encoder × 12 (pretrained BERT-base-cased → CoNLL-2003 NER fine-tuning)",
            fontsize=9.5, weight="bold", va="center")
    sub = ["Multi-head self-attention\n(12 heads, 64-d each)", "Add & LayerNorm", "Feed-forward\n768 → 3072 → 768 (GELU)", "Add & LayerNorm"]
    for i, s in enumerate(sub):
        box(ax, x0 + 0.2 + i * 2.38, 2.25, 2.2, 1.1, s, "#ffffff", fs=8.5)
        if i:
            arrow(ax, x0 + 0.2 + i * 2.38 - 0.18, 2.8, x0 + 0.2 + i * 2.38, 2.8)
    for i in range(n):
        arrow(ax, x0 + i * dx + 0.43, 0.75, x0 + i * dx + 0.43, 1.05, style="-")
    arrow(ax, 5.9, 1.65, 5.9, 2.0)
    box(ax, x0, 4.35, n * dx - 0.12, 0.55, "Contextual hidden states  h₁ … hₙ  (768-d per sub-token)", ORANGE)
    arrow(ax, 5.9, 4.0, 5.9, 4.35)
    # two heads
    box(ax, 0.3, 5.45, 5.2, 0.85, "BEFORE: original head  W ∈ ℝ^(9×768)\nPER / ORG / LOC / MISC  (B-, I-) + O", GREY, fs=8.5)
    box(ax, 6.2, 5.45, 5.5, 0.85, "AFTER: new head  W ∈ ℝ^(21×768), randomly initialised\n10 resume types (B-, I-) + O; whole network fine-tuned", GREEN, fs=8.5)
    arrow(ax, 4.0, 4.9, 2.9, 5.45); arrow(ax, 7.8, 4.9, 8.95, 5.45)
    labels = ["–", "B-NAME", "(−100)", "I-NAME", "B-DESIG.", "I-DESIG.", "O", "B-COMP.", "(−100)", "(−100)", "–"]
    ax.text(6.0, 6.75, "softmax per sub-token → label of the FIRST sub-token of each word (others ignored: −100 in training)",
            ha="center", fontsize=9, style="italic")
    for i, l in enumerate(labels):
        ax.text(x0 + i * dx + 0.43, 0.08, l, ha="center", va="top", fontsize=6.8, color="#2b8a3e")
    ax.text(x0 - 0.15, 0.08, "gold:", ha="right", va="top", fontsize=7, color="#2b8a3e")
    plt.savefig(FIGURES / "bert_ner_architecture.png", dpi=170, bbox_inches="tight"); plt.close()


def experiment_workflow():
    fig, ax = canvas(13, 5.2)
    box(ax, 0.2, 2.1, 2.2, 1.1, "DataTurks resumes\n220 JSON, char spans", ORANGE)
    box(ax, 2.9, 2.1, 2.4, 1.1, "data_prep.py\ndedup → clean → span→BIO\n219 resumes", GREY, fs=8.5)
    box(ax, 5.8, 2.1, 2.2, 1.1, "Split 70/15/15\n153 / 33 / 33\n+ 5-fold CV", GREY, fs=8.5)
    arrow(ax, 2.4, 2.65, 2.9, 2.65); arrow(ax, 5.3, 2.65, 5.8, 2.65)
    box(ax, 8.6, 3.9, 4.2, 1.05, "BEFORE  pretrained_eval.py\ndslim/bert-base-NER as-is, PER/ORG/LOC mapped", GREY, fs=8.5)
    box(ax, 8.6, 2.45, 4.2, 1.05, "AFTER  finetune.py\ngrid lr×bs → select on VAL F1 → final model", BLUE, fs=8.5)
    box(ax, 8.6, 1.0, 4.2, 1.05, "Ablations (5) + 5-fold CV\nepochs, class weights, oversampling,\ntruncation, normalisation, rules", PURPLE, fs=8.5)
    for y in (4.42, 2.97, 1.52):
        arrow(ax, 8.0, 2.65, 8.6, y)
    ax.text(10.7, 0.45, "Same test split · same metric: seqeval entity-level P/R/F1 (strict IOB2)",
            ha="center", fontsize=9, weight="bold", color="#364fc7")
    ax.text(6.5, 5.05, "Experimental design", ha="center", fontsize=11, weight="bold")
    plt.savefig(FIGURES / "experiment_workflow.png", dpi=170, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    system_architecture(); bert_architecture(); experiment_workflow()
    print("ok")
