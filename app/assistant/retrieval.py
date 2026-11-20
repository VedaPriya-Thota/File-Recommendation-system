"""
Passage-level retrieval over the user's indexed files.

  lexical  : BM25 (bm25s) over structure-aware passages (heading + text; the FILENAME is deliberately not part
             of the indexed text, so a misleading file name can never create evidence)
  semantic : the existing Qdrant index (one point per passage) when a real embedding model is installed
  fusion   : reciprocal-rank fusion, then reranking (cross-encoder if installed, otherwise a transparent
             term-coverage / proximity / phrase scorer), near-duplicate and per-file caps
  context  : neighbouring passages are attached so a hit keeps the sentences around it
  guard    : `assess()` decides whether the evidence really covers the question (term coverage + answer-type
             check) -- the basis for "I couldn't find that in your files"

Passage texts are never kept in memory by the index; only offsets are. Text is re-read from the database for the
few passages that are actually used.
"""
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import bm25s

from app.analysis import text as T
from app.assistant.understand import Need, synonym_stems
from app.db.connection import get_cursor
from app.ingestion.structure import Passage, build_passages

logger = logging.getLogger("file_discovery_agent.retrieval")

RRF_K = 60
_LOCK = threading.Lock()
_INDEX_CACHE: Dict[str, "PassageIndex"] = {}


@dataclass
class Ev:
    """One piece of evidence: a verbatim passage of a real file plus where it sits."""
    file_id: int
    idx: int
    start: int
    end: int
    text: str
    page: Optional[int] = None
    page_label: Optional[str] = None
    sheet: Optional[str] = None
    heading: Optional[str] = None
    line_start: int = 1
    line_end: int = 1
    score: float = 0.0
    coverage: float = 0.0
    need: int = 0
    before: str = ""
    after: str = ""
    sources: List[str] = field(default_factory=list)

    def location(self) -> str:
        return Passage(self.idx, self.start, self.end, self.page, self.page_label, self.sheet, self.heading, self.line_start, self.line_end).location()


@dataclass
class PRef:
    file_id: int
    idx: int
    start: int
    end: int
    page: Optional[int]
    page_label: Optional[str]
    sheet: Optional[str]
    heading: Optional[str]
    line_start: int
    line_end: int
    fp: Optional[str]


class PassageIndex:
    def __init__(self, refs: List[PRef], tokens: List[List[str]], signature):
        self.refs, self.signature = refs, signature
        self.by_file: Dict[int, List[int]] = {}
        for i, r in enumerate(refs):
            self.by_file.setdefault(r.file_id, []).append(i)
        self.n_files = len(self.by_file)
        self.bm25 = None
        if tokens and any(tokens):
            self.bm25 = bm25s.BM25()
            self.bm25.index(tokens, show_progress=False)

    def search(self, terms: Sequence[str], k: int) -> List[Tuple[int, float]]:
        if not self.bm25 or not terms or not self.refs:
            return []
        k = min(k, len(self.refs))
        res, scores = self.bm25.retrieve([list(terms)], k=k, show_progress=False)
        return [(int(i), float(s)) for i, s in zip(res[0], scores[0]) if s > 0]


def _ptokens(text: str, heading: Optional[str]) -> List[str]:
    joined = (heading + " " if heading else "") + text
    return [T._stem(t) for t in T.content_terms(re.sub(r"([a-z])([A-Z])", r"\1 \2", joined))]     # splitCamelCase identifiers


def _fetch_rows(owner_id: str, file_ids: Optional[Iterable[int]] = None) -> List[dict]:
    where, params = "f.owner_id = %(o)s AND s.state = 'content_indexed'", {"o": owner_id}
    if file_ids is not None:
        ids = list(file_ids)
        if not ids:
            return []
        where += " AND f.file_id = ANY(%(ids)s)"
        params["ids"] = ids
    with get_cursor() as cur:
        cur.execute(f"""SELECT f.file_id, f.file_type, f.extracted_text, s.meta_json, s.fingerprint
                        FROM files f JOIN file_ingestion_state s ON s.file_id = f.file_id
                        JOIN connected_folders c ON c.folder_id = s.folder_id
                        WHERE {where} ORDER BY f.file_id;""", params)
        return [dict(r) for r in cur.fetchall()]


def _build(owner_id: str, file_ids: Optional[Iterable[int]], signature) -> PassageIndex:
    refs: List[PRef] = []
    tokens: List[List[str]] = []
    for row in _fetch_rows(owner_id, file_ids):
        text = row["extracted_text"] or ""
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except ValueError:
            meta = {}
        for p in build_passages(text, meta, row["file_type"]):
            refs.append(PRef(row["file_id"], p.idx, p.start, p.end, p.page, p.page_label, p.sheet, p.heading, p.line_start, p.line_end, row["fingerprint"]))
            tokens.append(_ptokens(text[p.start:p.end], p.heading))
    return PassageIndex(refs, tokens, signature)


def _signature(owner_id: str):
    with get_cursor() as cur:
        cur.execute("""SELECT COUNT(*) AS n, COALESCE(SUM(f.file_id), 0) AS sid, MAX(f.updated_at) AS mu, MAX(s.indexed_at) AS mi, COALESCE(SUM(s.size_bytes), 0) AS sz
                       FROM files f JOIN file_ingestion_state s ON s.file_id = f.file_id
                       JOIN connected_folders c ON c.folder_id = s.folder_id
                       WHERE f.owner_id = %s AND s.state = 'content_indexed';""", (owner_id,))
        r = cur.fetchone()
    return (r["n"], int(r["sid"]), str(r["mu"]), str(r["mi"]), int(r["sz"]))


def get_index(owner_id: str, file_ids: Optional[Sequence[int]] = None) -> PassageIndex:
    """Whole-library index is cached and rebuilt only when the indexed files change; a scoped index (a handful
    of files) is built on the fly because that is cheaper than filtering a large one."""
    sig = _signature(owner_id)
    if file_ids:
        return _build(owner_id, file_ids, sig)
    with _LOCK:
        idx = _INDEX_CACHE.get(owner_id)
        if idx is not None and idx.signature == sig:
            return idx
    new = _build(owner_id, None, sig)
    with _LOCK:
        _INDEX_CACHE[owner_id] = new
    return new


# ---------------------------------------------------------------------------- scoring helpers
def _stem_set(text: str) -> set:
    return {T._stem(t) for t in T.tokens(re.sub(r"([a-z])([A-Z])", r"\1 \2", text))}


SYN_WEIGHT = 0.9     # a synonym satisfies a term, but an exact match is always preferred (see exact_coverage)


def coverage(need_stems: Sequence[str], text_stems: set) -> float:
    """Fraction of the question's key terms that the text speaks to (a synonym counts almost as much as the word)."""
    if not need_stems:
        return 0.0
    hit = 0.0
    for s in need_stems:
        if s in text_stems:
            hit += 1
        elif any(x in text_stems for x in synonym_stems(s)):
            hit += SYN_WEIGHT
    return hit / len(need_stems)


def exact_coverage(need_stems: Sequence[str], text_stems: set) -> float:
    return sum(1 for s in need_stems if s in text_stems) / len(need_stems) if need_stems else 0.0


def _proximity(need_stems: Sequence[str], text: str) -> float:
    """1.0 when all matched terms sit in one tight window, falling towards 0 as they spread out."""
    toks = [T._stem(t) for t in T.tokens(text)]
    pos = {}
    for i, t in enumerate(toks):
        for s in need_stems:
            if t in synonym_stems(s):
                pos.setdefault(s, []).append(i)
    if len(pos) < 2:
        return 0.5 if pos else 0.0
    best = min(max(p) - min(p) for p in _windows(pos))
    return 1.0 / (1.0 + best / 12.0)


def _windows(pos: Dict[str, List[int]]):
    keys = list(pos)
    anchor = keys[0]
    for a in pos[anchor][:40]:
        pick = [a]
        for k in keys[1:]:
            pick.append(min(pos[k], key=lambda x: abs(x - a)))
        yield pick


def _phrase_bonus(need_text: str, passage: str) -> float:
    qs = [T._stem(t) for t in T.content_terms(need_text)]
    ps = [T._stem(t) for t in T.content_terms(passage)]
    bigrams = {(a, b) for a, b in zip(qs, qs[1:])}
    if not bigrams:
        return 0.0
    got = sum(1 for bg in zip(ps, ps[1:]) if bg in bigrams)
    return min(1.0, got / len(bigrams))


def _semantic_real() -> bool:
    try:
        from app.embeddings import get_provider_name
        return not get_provider_name().startswith("hashing")
    except Exception:  # noqa: BLE001
        return False


def _cross_scores(query: str, texts: List[str]) -> Optional[List[float]]:
    try:
        from app.reranker import get_reranker_provider
        prov = get_reranker_provider()
        if prov.name.startswith("cross-encoder"):
            import math
            return [1 / (1 + math.exp(-x)) for x in prov.score(query, texts)]
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------- main retrieval
@dataclass
class Retrieval:
    evidence: List[Ev]
    searched_files: int
    searched_passages: int
    steps: List[str]
    channels: List[str]


def _load_texts(owner_id: str, file_ids: Iterable[int]) -> Dict[int, str]:
    ids = list(set(file_ids))
    if not ids:
        return {}
    with get_cursor() as cur:
        cur.execute("SELECT file_id, extracted_text FROM files WHERE owner_id = %(o)s AND file_id = ANY(%(ids)s);", {"o": owner_id, "ids": ids})
        return {r["file_id"]: r["extracted_text"] or "" for r in cur.fetchall()}


def retrieve_need(owner_id: str, need: Need, need_no: int, scope_ids: Optional[Sequence[int]] = None, top_n: int = 4,
                  per_file_cap: int = 3, pool: int = 30, relaxed: bool = False) -> Retrieval:
    idx = get_index(owner_id, scope_ids)
    steps, channels = [], ["bm25"]
    query_stems = list(dict.fromkeys(s for st in need.stems for s in synonym_stems(st)))
    fused: Dict[int, float] = {}
    srcs: Dict[int, List[str]] = {}

    lex = idx.search(query_stems, pool)
    for rank, (i, _) in enumerate(lex):
        fused[i] = fused.get(i, 0) + 1.0 / (RRF_K + rank)
        srcs.setdefault(i, []).append("keyword")
    dense_n = 0
    if _semantic_real() and need.terms:
        channels.append("semantic")
        try:
            from app.tools import semantic_search as sem
            ref_key = {(r.file_id, r.idx): i for i, r in enumerate(idx.refs)}
            for rank, h in enumerate(sem.search_chunks(need.text if len(need.terms) > 1 else need.query, [owner_id], pool, scope_ids or None)):
                i = ref_key.get((h["file_id"], h["chunk_id"]))
                if i is not None and idx.refs[i].start == h.get("start"):
                    fused[i] = fused.get(i, 0) + 1.0 / (RRF_K + rank)
                    srcs.setdefault(i, []).append("meaning")
                    dense_n += 1
        except Exception as exc:  # noqa: BLE001 -- semantic channel is optional; keyword channel already ran
            steps.append(f"semantic search unavailable ({type(exc).__name__})")
    if not fused:
        return Retrieval([], idx.n_files, len(idx.refs), steps, channels)

    cand = sorted(fused, key=lambda i: -fused[i])[:pool]
    texts = _load_texts(owner_id, {idx.refs[i].file_id for i in cand})
    ptexts = {i: texts.get(idx.refs[i].file_id, "")[idx.refs[i].start:idx.refs[i].end] for i in cand}
    cross = _cross_scores(need.text, [ptexts[i] for i in cand]) if len(cand) > 1 else None
    if cross:
        channels.append("reranker")
    top_rrf = max(fused.values())
    scored: List[Tuple[float, int, float]] = []
    for pos, i in enumerate(cand):
        r = idx.refs[i]
        tstems = _stem_set((r.heading or "") + " " + ptexts[i])
        cov = coverage(need.stems, tstems)
        prox = _proximity(need.stems, ptexts[i])
        phrase = _phrase_bonus(need.text, ptexts[i])
        base = 0.55 * cov + 0.15 * prox + 0.10 * phrase + 0.20 * (fused[i] / top_rrf)
        if cross:
            base = 0.5 * base + 0.5 * cross[pos]
        scored.append((base, i, cov))
    scored.sort(key=lambda x: -x[0])

    chosen: List[Ev] = []
    per_file: Dict[int, int] = {}
    seen_fp: set = set()
    for base, i, cov in scored:
        r = idx.refs[i]
        if per_file.get(r.file_id, 0) >= per_file_cap:
            continue
        if (r.fp, r.idx) in seen_fp:                      # byte-identical copy of an already chosen passage
            continue
        if any(e.file_id == r.file_id and not (r.end <= e.start or r.start >= e.end) for e in chosen):
            continue
        if cov == 0 and not relaxed:
            continue
        seen_fp.add((r.fp, r.idx))
        per_file[r.file_id] = per_file.get(r.file_id, 0) + 1
        full = texts.get(r.file_id, "")
        ev = Ev(file_id=r.file_id, idx=r.idx, start=r.start, end=r.end, text=full[r.start:r.end], page=r.page, page_label=r.page_label,
                sheet=r.sheet, heading=r.heading, line_start=r.line_start, line_end=r.line_end, score=round(base, 4), coverage=round(cov, 3),
                need=need_no, sources=srcs.get(i, []))
        chosen.append(ev)
        if len(chosen) >= top_n:
            break

    # neighbouring passages as context (kept separate from the evidence text itself)
    for ev in chosen:
        sib = idx.by_file.get(ev.file_id, [])
        try:
            pos = [idx.refs[j].idx for j in sib].index(ev.idx)
        except ValueError:
            continue
        full = texts.get(ev.file_id, "")
        if pos > 0:
            pr = idx.refs[sib[pos - 1]]
            ev.before = full[pr.start:pr.end][-500:]
        if pos + 1 < len(sib):
            nr = idx.refs[sib[pos + 1]]
            ev.after = full[nr.start:nr.end][:500]
    steps.append(f"{len(lex)} keyword hit(s)" + (f", {dense_n} by meaning" if dense_n else "") + f" over {len(idx.refs)} passages in {idx.n_files} file(s)")
    return Retrieval(chosen, idx.n_files, len(idx.refs), steps, channels)


# ---------------------------------------------------------------------------- sufficiency
_NUMBER = re.compile(r"\d")
_NUMWORDS = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|billion|half|dozen|none|zero)\b", re.I)
_DATEISH = re.compile(r"\b(\d{1,2}(:\d{2})?\s?(am|pm|a\.m\.|p\.m\.)|noon|midnight|morning|afternoon|evening|daily|weekly|monthly|annually|quarterly|every \w+|weekends?|weekdays?|business days?|jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                      r"q[1-4]|quarter|today|tomorrow|yesterday|(19|20)\d{2}|\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?)\b", re.I)
_BECAUSE = re.compile(r"\b(because|due to|since|as a result|so that|in order to|owing to|caused by|reason|therefore|leads? to|led to)\b", re.I)


def type_ok(answer_type: str, sentence: str) -> bool:
    if answer_type == "number":
        return bool(_NUMBER.search(sentence) or _NUMWORDS.search(sentence))
    if answer_type == "date":
        return bool(_DATEISH.search(sentence))
    if answer_type == "reason":
        return bool(_BECAUSE.search(sentence))
    if answer_type == "person":
        return bool(re.search(r"\b[A-Z][a-z]+(\s+[A-Z][a-z]+)+\b|\b(he|she|team|manager|director|owner|lead|ceo|cto|cfo)\b", sentence))
    return True


def need_threshold(n_terms: int) -> float:
    if n_terms <= 2:
        return 0.9
    if n_terms <= 4:
        return 0.66
    return 0.6


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    """(start, end, sentence) with offsets into `text`; list items and table rows count as sentences."""
    out = []
    pos = 0
    for m in re.finditer(r"[^\n]+", text):
        line, base = m.group(0), m.start()
        parts, cur = [], 0
        for sm in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", line):
            parts.append((cur, sm.start()))
            cur = sm.end()
        parts.append((cur, len(line)))
        for a, b in parts:
            s = line[a:b].strip()
            if len(s) >= 12 or (s and any(c.isdigit() for c in s)):
                lead = len(line[a:b]) - len(line[a:b].lstrip())
                out.append((base + a + lead, base + a + lead + len(s), s))
    return out


def best_sentences(need: Need, evs: List[Ev], limit: int = 3) -> List[dict]:
    """Rank the sentences inside the retrieved passages (and their neighbours) by how directly they address the need."""
    out = []
    for ev in evs:
        for a, b, s in split_sentences(ev.text):
            st = _stem_set(s)
            hstems = _stem_set(ev.heading) if ev.heading else set()
            cov_s = coverage(need.stems, st)
            cov = coverage(need.stems, st | hstems)          # a sentence under a matching heading answers "what are the limitations"
            via_heading = cov > cov_s
            if cov == 0:
                continue
            from app.ingestion.structure import is_title_line
            if (ev.heading and s.strip().lstrip("#").strip().rstrip(":") == ev.heading) or (is_title_line(s) and not any(ch.isdigit() for ch in s)):
                continue                                   # a heading/title is a label, not an answer
            if a == 0 and len(s.split()) <= 9 and not re.search(r"[.!?:;]$", s) and len(ev.text) > len(s) + 20 and not any(ch.isdigit() for ch in s):
                continue                                   # the first line of a passage without a verb-like ending is a title
            if len(s.split()) < 4 and not (any(ch.isdigit() for ch in s) and need.answer_type in ("number", "date")):
                continue                                   # titles / labels are not statements
            ok = type_ok(need.answer_type, s)
            heading_bonus = 0.1 if ev.heading and coverage(need.stems, _stem_set(ev.heading)) > 0 else 0.0
            score = cov + 0.3 * exact_coverage(need.stems, st | hstems) + (0.25 if ok else -0.35 if need.answer_type in ("number", "date") else 0.0) + 0.2 * ev.score + heading_bonus - (0.15 if len(s) > 400 else 0)
            if via_heading:
                score -= 0.05
            out.append({"sentence": s, "start": ev.start + a, "end": ev.start + b, "ev": ev, "cov": cov, "type_ok": ok, "score": score, "via_heading": via_heading and cov_s == 0})
    out.sort(key=lambda x: -x["score"])
    chosen, seen = [], []
    for c in out:
        toks = set(T.stems(c["sentence"])) | {"#" + n for n in re.findall(r"\d[\d,.]*", c["sentence"])}   # sentences that differ in a figure are NOT duplicates
        if any(len(toks & o) / max(len(toks | o), 1) > 0.7 for o in seen):
            continue
        seen.append(toks)
        chosen.append(c)
        if len(chosen) >= limit:
            break
    return chosen


def assess(need: Need, evs: List[Ev]) -> dict:
    """Is there real evidence for this need? -> {ok, best, reason, missing_terms}."""
    if not need.stems:
        return {"ok": False, "best": None, "reason": "no_terms", "missing": []}
    thr = need_threshold(len(need.stems))
    cands = best_sentences(need, evs, 5)
    for c in cands:
        local_cov = c["cov"]
        if local_cov >= thr and c["type_ok"]:
            return {"ok": True, "best": c, "reason": "ok", "missing": []}
    # a sentence may only carry part of the topic while its passage (or heading) carries the rest
    for c in cands:
        ev = c["ev"]
        joint = coverage(need.stems, _stem_set((ev.heading or "") + " " + ev.text))
        if joint >= thr and c["cov"] >= 0.5 * thr and c["type_ok"]:
            return {"ok": True, "best": c, "reason": "ok_passage", "missing": []}
    seen_stems = set()
    for ev in evs:
        seen_stems |= _stem_set(ev.text)
    missing = [t for t, s in zip(need.terms, need.stems) if not any(x in seen_stems for x in synonym_stems(s))]
    reason = "wrong_type" if cands and any(c["cov"] >= thr for c in cands) else "low_coverage"
    return {"ok": False, "best": cands[0] if cands else None, "reason": reason, "missing": missing}
