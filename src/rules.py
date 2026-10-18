"""Regex rule layer applied on top of model predictions (Phase 3 ablation, also used in the app).
  - EMAIL: any e-mail address in the text is forced to EMAIL (overrides the model).
  - GRADUATION_YEAR: a 4-digit year on the same line as a predicted DEGREE or COLLEGE_NAME,
    and not already inside an entity, is tagged GRADUATION_YEAR.
  - PHONE: extracted for the API output only (not an annotated label, so not evaluated).
"""
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
                      r"|indeed\.com/r/[A-Za-z0-9\-]+/[0-9a-f]+")
YEAR_RE = re.compile(r"\b(19[6-9]\d|20[0-4]\d)\b")
PHONE_RE = re.compile(r"(?<![\w/])\+?\(?\d[\d\s\-()]{8,16}\d(?![\w/])")


def _mark(tags, idx, typ):
    for k, i in enumerate(idx):
        tags[i] = ("B-" if k == 0 else "I-") + typ


def apply_rules(text, starts, ends, tags):
    tags = list(tags)
    # EMAIL
    for m in EMAIL_RE.finditer(text):
        idx = [i for i, (s, e) in enumerate(zip(starts, ends)) if s >= m.start() and e <= m.end()]
        if idx:
            _mark(tags, idx, "EMAIL")
    # GRADUATION_YEAR
    line_of = []
    line = 0
    pos = 0
    for s in starts:
        line += text.count("\n", pos, s)
        pos = s
        line_of.append(line)
    edu_lines = {line_of[i] for i, t in enumerate(tags) if t[2:] in ("DEGREE", "COLLEGE_NAME")}
    for i, (s, e) in enumerate(zip(starts, ends)):
        if tags[i] == "O" and YEAR_RE.fullmatch(text[s:e]) and line_of[i] in edu_lines:
            tags[i] = "B-GRADUATION_YEAR"
    return tags


def find_phones(text):
    return [m.group().strip() for m in PHONE_RE.finditer(text)
            if 10 <= sum(c.isdigit() for c in m.group()) <= 13 and "\n" not in m.group()
            and not all(YEAR_RE.fullmatch(g) for g in re.findall(r"\d+", m.group()))]
