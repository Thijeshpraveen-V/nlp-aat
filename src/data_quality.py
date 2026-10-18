"""Quantify annotation-quality issues in the gold data (used in report sections 5 and 8).
Writes results/data_quality.json."""
import re
from collections import Counter

from common import DATA_PROCESSED, ENTITY_TYPES, RESULTS, load_json, read_jsonl, save_json

docs = read_jsonl(DATA_PROCESSED / "clean" / "docs.jsonl")
URL = re.compile(r"indeed\.com/r/")
AT = re.compile(r"@")

span_text = lambda d, s: d["text"][s[0]:s[1]]
gold = [(d, s) for d in docs for s in d["spans"]]

email = [span_text(d, s) for d, s in gold if s[2] == "EMAIL"]
urls_by_type = Counter(s[2] for d, s in gold if URL.search(span_text(d, s)))
docs_with_url = sum(1 for d in docs if URL.search(d["text"]))
docs_url_unlabelled = sum(1 for d in docs if URL.search(d["text"]) and not any(
    URL.search(span_text(d, s)) for s in d["spans"]))

# entity types never annotated in a resume although that resume has the section heading
heads = {"SKILLS": r"\bSKILLS\b", "COLLEGE_NAME": r"\bEDUCATION\b", "COMPANIES_WORKED_AT": r"\bWORK EXPERIENCE\b"}
section_unlabelled = {}
for t, h in heads.items():
    with_head = [d for d in docs if re.search(h, d["text"])]
    section_unlabelled[t] = {"resumes_with_heading": len(with_head),
                             "of_which_no_gold_span": sum(1 for d in with_head if not any(s[2] == t for s in d["spans"]))}

lens = {t: [len(span_text(d, s).split()) for d, s in gold if s[2] == t] for t in ENTITY_TYPES}
docs_per_type = {t: sum(1 for d in docs if any(s[2] == t for s in d["spans"])) for t in ENTITY_TYPES}

# fraction of fine-tuned "spurious" predictions whose exact text is annotated elsewhere in the same resume
final = load_json(RESULTS / "final_config.json")["final_run"]
err = load_json(RESULTS / "error_analysis.json")

out = {
    "n_resumes": len(docs),
    "email_spans": len(email),
    "email_spans_indeed_url": sum(1 for e in email if URL.search(e)),
    "email_spans_with_at": sum(1 for e in email if AT.search(e)),
    "indeed_url_spans_by_gold_type": dict(urls_by_type),
    "resumes_containing_indeed_url": docs_with_url,
    "resumes_with_url_not_annotated": docs_url_unlabelled,
    "section_heading_but_no_annotation": section_unlabelled,
    "resumes_with_at_least_one_span_of_type": docs_per_type,
    "span_length_words": {t: {"mean": sum(v) / len(v), "max": max(v)} for t, v in lens.items() if v},
}
save_json(RESULTS / "data_quality.json", out)
import json
print(json.dumps(out, indent=1))
