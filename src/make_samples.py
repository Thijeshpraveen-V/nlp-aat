"""Create demo inputs in data/sample_resumes/:
  - 4 resumes from the held-out TEST split as .txt + .gold.json (gold entities for the demo)
  - 1 fictional resume written for this project, as .pdf and .docx (exercises the file pipeline)
"""
import matplotlib

matplotlib.use("Agg")
import docx
import matplotlib.pyplot as plt

from common import ROOT, load_split, save_json

OUT = ROOT / "data" / "sample_resumes"

FICTIONAL = """Ananya Rao
Data Scientist - Fintrix Analytics
Mysuru, Karnataka
Email: ananya.rao.demo@example.com | Phone: +91 98765 43210

WORK EXPERIENCE
Data Scientist
Fintrix Analytics - Bengaluru, Karnataka
June 2021 to Present
Built credit-risk models and NLP pipelines for document classification.

Junior Analyst
Northwind Retail Solutions - Chennai, Tamil Nadu
July 2019 to May 2021

EDUCATION
M.Tech in Computer Science
National Institute of Engineering, Mysuru
2019

B.E. in Information Science
Sri Jayachamarajendra College of Engineering
2017

SKILLS
Python, PyTorch, SQL, Machine Learning, Natural Language Processing, Docker, AWS
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits, _ = load_split("clean")
    cands = [d for d in splits["test"] if len(d["tokens"]) < 900]
    cands.sort(key=lambda d: (-len({t for _, _, t in d["spans"]}), d["id"]))
    for d in cands[:4]:
        (OUT / f"test_{d['id']}.txt").write_bytes(d["text"].encode("utf-8"))  # LF only, so gold offsets match
        save_json(OUT / f"test_{d['id']}.gold.json",
                  [{"text": d["text"][s:e], "label": t, "start": s, "end": e} for s, e, t in d["spans"]])

    doc = docx.Document()
    for line in FICTIONAL.splitlines():
        doc.add_paragraph(line)
    doc.save(OUT / "sample_fictional.docx")

    lines = FICTIONAL.splitlines()
    fig = plt.figure(figsize=(8.27, 11.69))
    for i, line in enumerate(lines):
        fig.text(0.08, 0.95 - i * 0.026, line, fontsize=10, family="DejaVu Sans",
                 weight="bold" if line.isupper() or i == 0 else "normal")
    fig.savefig(OUT / "sample_fictional.pdf")
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
