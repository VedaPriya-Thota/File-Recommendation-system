"""Deterministic text analysis over REAL extracted text: passage ranking, excerpts with verifiable offsets,
extractive summaries and document comparison. No model calls."""
import math
import re
from collections import Counter
from typing import Dict, List, Sequence

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
STOP = set("""a an the and or but if then else of to in on at by for from with without about as is are was were be been being it its this
that these those i me my we our you your he she they them their his her not no nor so than too very can could should would may might will
shall do does did done have has had having also into over under between among per via etc such there here which who whom whose what when where
why how can't won't don't doesn't isn't aren't wasn't weren't couldn't shouldn't wouldn't didn't haven't hasn't all any both each few more most other some own same just only up down out off again further once""".split())


def tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def content_terms(text: str) -> List[str]:
    return [t for t in tokens(text) if t not in STOP and len(t) > 1]


def query_terms(query: str, extra: Sequence[str] = ()) -> List[str]:
    out, seen = [], set()
    for t in content_terms(query) + [x.lower() for x in extra]:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _stem(t: str) -> str:
    """Light, deterministic suffix stripper (same function for index and query side)."""
    t = t.lower().replace("’", "'").strip("'-")
    if t.endswith("'s"):
        t = t[:-2]
    t = t.replace("'", "")
    if len(t) <= 3 or t.isdigit():
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("sses"):
        return t[:-2]
    for suf in ("ations", "ation", "ments", "ment", "ness", "ings", "ing", "ed", "es", "ly", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[: -len(suf)]
            break
    if len(t) > 4 and t[-1] == t[-2] and t[-1] not in "ls":      # running -> runn -> run
        t = t[:-1]
    return t.rstrip("e") if len(t) > 4 else t


def stems(text: str) -> List[str]:
    return [_stem(t) for t in content_terms(text)]


def split_passages(text: str, target: int = 450) -> List[Dict]:
    """Paragraph/sentence-aligned windows with exact character offsets into `text`."""
    passages, pos = [], 0
    for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        para, start = m.group(0), m.start()
        if len(para) <= target * 1.5:
            passages.append({"start": start, "end": m.end()})
            continue
        # long paragraph: cut at sentence boundaries
        cur = start
        for sm in re.finditer(r"(?<=[.!?])\s+", para):
            if start + sm.start() - cur >= target:
                passages.append({"start": cur, "end": start + sm.start()})
                cur = start + sm.end()
        if cur < m.end():
            passages.append({"start": cur, "end": m.end()})
    return [p for p in passages if text[p["start"]:p["end"]].strip()]


def top_excerpts(text: str, terms: Sequence[str], n: int = 3, max_chars: int = 500) -> List[Dict]:
    """Best-matching passages for the query terms. Every excerpt is `text[start:end]` verbatim (verifiable)."""
    if not text:
        return []
    stems = [_stem(t) for t in terms if t]
    passages = split_passages(text, target=max_chars // 1.2 if max_chars > 200 else 200)
    n_pass = max(len(passages), 1)
    df = Counter()
    toks_by = []
    for p in passages:
        ts = {_stem(t) for t in tokens(text[p["start"]:p["end"]])}
        toks_by.append(ts)
        for s in stems:
            if s in ts:
                df[s] += 1
    scored = []
    for p, ts in zip(passages, toks_by):
        hit = [s for s in set(stems) if s in ts]
        if not hit:
            continue
        idf = sum(math.log(1 + n_pass / (1 + df[s])) for s in hit)
        cover = len(hit) / max(len(set(stems)), 1)
        scored.append((idf * (0.5 + cover), p, hit))
    scored.sort(key=lambda x: (-x[0], x[1]["start"]))
    out, used = [], []
    for score, p, hit in scored:
        if any(not (p["end"] <= a or p["start"] >= b) for a, b in used):
            continue
        s, e = p["start"], min(p["end"], p["start"] + max_chars)
        if e < p["end"]:
            cut = text.rfind(" ", s, e)
            e = cut if cut > s + 40 else e
        out.append({"text": text[s:e], "start": s, "end": e, "score": round(score, 3), "matched_terms": sorted(hit)})
        used.append((p["start"], p["end"]))
        if len(out) >= n:
            break
    return out


def head_excerpt(text: str, max_chars: int = 500) -> Dict:
    t = text[:max_chars]
    cut = t.rfind(" ") if len(text) > max_chars else len(t)
    e = cut if cut > 40 else len(t)
    return {"text": text[:e], "start": 0, "end": e, "score": 0.0, "matched_terms": []}


def term_counts(text: str, terms: Sequence[str]) -> Dict[str, int]:
    stem_counts = Counter(_stem(t) for t in tokens(text))
    return {t: stem_counts.get(_stem(t), 0) for t in terms if stem_counts.get(_stem(t), 0)}


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def extractive_summary(text: str, n: int = 5) -> List[Dict]:
    """Frequency-weighted sentence extraction (position-aware). Returns verbatim sentences in document order."""
    sents = sentences(text)
    if not sents:
        return []
    freq = Counter(_stem(t) for t in content_terms(text))
    if not freq:
        return [{"text": s} for s in sents[:n]]
    top = max(freq.values())
    scored = []
    for i, s in enumerate(sents):
        ts = [_stem(t) for t in content_terms(s)]
        if not ts:
            continue
        score = sum(freq[t] / top for t in ts) / math.sqrt(len(ts))
        if i < 3:
            score *= 1.15
        scored.append((score, i, s))
    chosen = sorted(sorted(scored, reverse=True)[:n], key=lambda x: x[1])
    return [{"text": s, "position": i} for _, i, s in chosen]


def keywords(text: str, k: int = 12) -> List[str]:
    c = Counter(content_terms(text))
    return [w for w, _ in c.most_common(k)]


_NUM = re.compile(r"(?<![\w.])[$€£]?\d[\d,]*(?:\.\d+)?%?")


def numbers(text: str) -> List[str]:
    return [n for n in _NUM.findall(text) if len(n.strip("$€£%,")) >= 2]


def headings(text: str, limit: int = 25) -> List[str]:
    hs = []
    for line in text.split("\n"):
        l = line.strip()
        if not l or len(l) > 80:
            continue
        if l.startswith("#") or (l.isupper() and len(l) > 3) or re.match(r"^(\d+(\.\d+)*[.)]?|[IVX]+\.)\s+\S", l) or l.startswith("[Slide") or l.startswith("[Sheet"):
            hs.append(l.lstrip("# ").strip())
        if len(hs) >= limit:
            break
    return hs


def compare_documents(a: str, b: str) -> Dict:
    ta, tb = Counter(content_terms(a)), Counter(content_terms(b))
    va, vb = set(ta), set(tb)
    inter = va & vb
    jacc = len(inter) / max(len(va | vb), 1)
    dot = sum(ta[t] * tb[t] for t in inter)
    na = math.sqrt(sum(v * v for v in ta.values())) or 1
    nb = math.sqrt(sum(v * v for v in tb.values())) or 1
    shared = sorted(inter, key=lambda t: -(ta[t] + tb[t]))[:12]
    only_a = sorted(va - vb, key=lambda t: -ta[t])[:10]
    only_b = sorted(vb - va, key=lambda t: -tb[t])[:10]
    na_set, nb_set = set(numbers(a)), set(numbers(b))
    ha, hb = headings(a), headings(b)
    return {
        "similarity": round(dot / (na * nb), 3), "vocabulary_overlap": round(jacc, 3),
        "words": [len(tokens(a)), len(tokens(b))], "shared_terms": shared, "only_in_first": only_a, "only_in_second": only_b,
        "numbers_only_in_first": sorted(na_set - nb_set)[:12], "numbers_only_in_second": sorted(nb_set - na_set)[:12],
        "numbers_in_both": sorted(na_set & nb_set)[:12],
        "headings_first": ha[:12], "headings_second": hb[:12],
        "headings_only_in_first": [h for h in ha if h.lower() not in {x.lower() for x in hb}][:10],
        "headings_only_in_second": [h for h in hb if h.lower() not in {x.lower() for x in ha}][:10],
    }
