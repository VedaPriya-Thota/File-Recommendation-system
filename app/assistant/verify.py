"""
Answer verification. Every factual sentence of an answer is checked against the passages it cites (or, if it
cites nothing, against all retrieved evidence):

  * numbers, percentages, dates and quotations in the sentence must literally appear in the evidence
  * enough of the sentence's content words must be supported by the evidence
  * citation markers must refer to real sources

Unsupported sentences are removed (and reported), supported sentences that lack a marker get one. The result is
what the user sees; the report is returned for transparency.
"""
import re
from typing import Dict, List, Tuple

from app.analysis import text as T

_CITE = re.compile(r"\[(\d+)(?:\s*·\s*[^\]]*)?\]")
_NUMTOK = re.compile(r"(?<![\w.])[$€£]?\d[\d,]*(?:\.\d+)?%?")
_QUOTE = re.compile(r"[“\"]([^”\"\n]{20,300})[”\"]")
_FILLER = re.compile(r"^(here('s| is| are)|in short|overall|to summari[sz]e|i (found|could|couldn't|can't|searched|looked|got this|then re-computed)|based on|according to|note|unfortunately|sorry|"
                     r"the (files|documents|sources) (don't|do not)|there (is|are) no|it looks like|this looks like|taken together|this is based on|this comes from|what the page|the main points|related code|"
                     r"settings found|libraries (it relies|involved)|a data note|your files disagree|check which|related:|\(that last part|the nearest material|if you expect)\b", re.I)


def _nums(text: str) -> set:
    out = set()
    for m in _NUMTOK.findall(text):
        n = re.sub(r"[$€£,%]", "", m).rstrip(".")
        out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def _sentences(md: str) -> List[Tuple[str, str]]:
    """[(prefix, sentence)] keeping bullet/heading prefixes so markdown structure survives."""
    out = []
    for line in md.split("\n"):
        m = re.match(r"^(\s*(?:[-*•]|\d+[.)])\s+|#{1,4}\s+)?(.*)$", line)
        prefix, body = m.group(1) or "", m.group(2)
        if not body.strip():
            out.append((prefix, ""))
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(\[*])", body)
        for i, p in enumerate(parts):
            out.append((prefix if i == 0 else "", p))
    return out


def _supported(sentence: str, ev_stems: set, ev_nums: set, ev_text_norm: str) -> Tuple[float, bool]:
    clean = _CITE.sub("", sentence)
    stems = [T._stem(t) for t in T.content_terms(clean)]
    if not stems:
        return 1.0, True
    frac = sum(1 for s in stems if s in ev_stems) / len(stems)
    nums_ok = all(n in ev_nums for n in _nums(clean) if len(n.replace(".", "")) >= 1 and not (len(n) == 1 and n in "0123456789" and False))
    quotes_ok = all(re.sub(r"\s+", " ", q).strip().lower() in ev_text_norm for q in _QUOTE.findall(clean))
    return frac, nums_ok and quotes_ok


def verify_answer(answer: str, sources: Dict[int, List[str]], extra_facts: str = "", min_support: float = 0.55) -> Tuple[str, dict]:
    """sources: {n: [passage texts]} (n = citation number)."""
    all_text = " ".join(t for v in sources.values() for t in v) + " " + extra_facts
    by_src_stems = {n: {T._stem(t) for t in T.tokens(" ".join(v))} for n, v in sources.items()}
    by_src_nums = {n: _nums(" ".join(v)) for n, v in sources.items()}
    all_stems = {T._stem(t) for t in T.tokens(all_text)}
    all_nums = _nums(all_text)
    all_norm = re.sub(r"\s+", " ", all_text).lower()
    report = {"sentences_checked": 0, "supported": 0, "removed": [], "citations_added": 0, "invalid_citations_removed": []}
    out_lines: List[str] = []
    cur_line_parts: List[str] = []
    prefix_for_line = ""
    pending: List[Tuple[str, str]] = _sentences(answer)
    # rebuild line by line: _sentences emits one entry per sentence; empty body = blank line
    lines: List[str] = []
    buf: List[str] = []
    buf_prefix = ""
    for prefix, sent in pending:
        if not sent.strip():
            if buf:
                lines.append(buf_prefix + " ".join(buf))
                buf = []
            lines.append("")
            buf_prefix = ""
            continue
        if prefix and buf:
            lines.append(buf_prefix + " ".join(buf))
            buf = []
        if prefix or not buf:
            buf_prefix = prefix if prefix else buf_prefix
        # --- check this sentence
        cites = [int(c) for c in _CITE.findall(sent)]
        bad = [c for c in cites if c not in sources]
        for b in bad:
            sent = re.sub(rf"\[{b}(?:\s*·\s*[^\]]*)?\]", "", sent)
            report["invalid_citations_removed"].append(b)
        cites = [c for c in cites if c in sources]
        factual = bool(T.content_terms(_CITE.sub("", sent))) and not _FILLER.match(sent.strip().lstrip("*"))
        if factual and len(T.content_terms(_CITE.sub("", sent))) >= 3:
            report["sentences_checked"] += 1
            if cites:
                ev_stems = set().union(*(by_src_stems[c] for c in cites))
                ev_nums = set().union(*(by_src_nums[c] for c in cites)) | _nums(extra_facts)
                frac, ok = _supported(sent, ev_stems, ev_nums, re.sub(r"\s+", " ", " ".join(" ".join(sources[c]) for c in cites)).lower())
                if not (ok and frac >= min_support):          # cited source does not support it: maybe another does
                    frac2, ok2 = _supported(sent, all_stems, all_nums, all_norm)
                    if ok2 and frac2 >= min_support:
                        ok, frac = True, frac2
            else:
                frac, ok = _supported(sent, all_stems, all_nums, all_norm)
            if not (ok and frac >= min_support):
                report["removed"].append(_CITE.sub("", sent).strip()[:160])
                continue
            report["supported"] += 1
            if not cites and len(sources) >= 1:
                best = max(sources, key=lambda n: len({T._stem(t) for t in T.content_terms(sent)} & by_src_stems[n]))
                sent = sent.rstrip() + f" [{best}]"
                report["citations_added"] += 1
        buf.append(sent)
    if buf:
        lines.append(buf_prefix + " ".join(buf))
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned, report
