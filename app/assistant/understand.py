"""
Question understanding. Turns a user message (+ conversation context) into a QuestionPlan BEFORE anything is
retrieved: what kind of question is it, which files is it about, which pieces of information are needed to
answer it, and what an answer must look like (a number? a date? a summary?).

Deterministic on purpose (no model call): it is fast, testable, and the retrieval stage must not depend on an
optional LLM being configured. The plan also records follow-up resolution ("its limitations" -> the report the
previous answer was about) so that retrieval can stay narrow and targeted.
"""
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.analysis import text as T
from app.assistant.intent import classify as classify_find
from app.db.connection import get_cursor

# ---------------------------------------------------------------------------- vocabulary
FLUFF = set("""tell show say says said mention mentions mentioned according given provide provides please explain describe know find want need get
give using used within connected folder folders file files document documents doc docs pdf report reports text content contents information info
detail details something anything everything thing things stuff kind sort type exactly actually basically really also still just much many analyze analyse identify csv xlsx excel spreadsheet spreadsheets sheet function functions method methods class classes module modules script scripts code main key major important biggest happen happened happens occur occurred went going being become became""".split())

SYNONYMS: Dict[str, List[str]] = {
    "limitation": ["limitation", "drawback", "constraint", "caveat", "weakness", "shortcoming", "downside", "restriction", "future work", "not able", "cannot"],
    "finding": ["finding", "result", "conclusion", "outcome", "discovered", "showed", "found that", "observed"],
    "recommendation": ["recommendation", "recommend", "suggest", "should", "advise", "propose"],
    "risk": ["risk", "threat", "issue", "concern", "challenge", "danger", "vulnerability"],
    "method": ["method", "methodology", "approach", "procedure", "technique", "design"],
    "cost": ["cost", "budget", "expense", "spend", "expenditure"],
    "deadline": ["deadline", "due", "due date", "timeline", "schedule", "milestone", "by"],
    "revenue": ["revenue", "sales", "income", "turnover", "earnings"],
    "goal": ["goal", "objective", "aim", "purpose", "mission", "target"],
    "benefit": ["benefit", "advantage", "pro", "strength", "upside"],
    "employee": ["employee", "staff", "personnel", "headcount", "worker", "team member"],
    "policy": ["policy", "rule", "guideline", "regulation", "requirement"],
    "refund": ["refund", "reimbursement", "money back", "return"],
    "train": ["train", "training", "trained", "fit", "epoch", "optimizer", "loss", "backward", "gradient", "learning rate"],
    "predict": ["predict", "prediction", "inference", "infer", "evaluate", "score"],
    "data": ["data", "dataset", "dataloader", "load", "preprocess", "feature"],
}
_SYN_INDEX: Dict[str, str] = {}
for _k, _vs in SYNONYMS.items():
    for _v in _vs:
        for _w in _v.split():
            _SYN_INDEX.setdefault(T._stem(_w), _k)


def synonym_stems(term_stem: str) -> List[str]:
    key = _SYN_INDEX.get(term_stem)
    if not key:
        return [term_stem]
    out = {term_stem}
    for v in SYNONYMS[key]:
        if " " not in v:
            out.add(T._stem(v))
    return sorted(out)


_FOLLOWUP = re.compile(
    r"\b(it|its|it's|this|that|these|those|they|them|their|the (same )?(report|document|file|paper|spreadsheet|sheet|code|project|one)|"
    r"the above|previous|earlier|same (one|file|document|report))\b", re.I)
_FOLLOWUP_START = re.compile(r"^\s*(and|also|what about|how about|but|so|then|why|how come|can you (also )?(explain|elaborate|expand)|more (on|about|detail)|elaborate|go on|continue)\b", re.I)
_WH = re.compile(r"^\s*(what|which|who|whom|whose|when|where|why|how|is|are|was|were|does|do|did|can|could|should|will|would|has|have|had)\b", re.I)

_RE_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|differences?|differ|differs|contrast|similarities|similar to|better than|which is (better|cheaper|higher|larger|bigger)|how (does|do|is|are) .{0,60}(differ|compare))\b", re.I)
_RE_SUMMARY = re.compile(r"\b(summari[sz]e|summary|tl;?dr|overview|main points?|key (points?|takeaways?|findings?|results?)|main (findings?|ideas?|conclusions?|results?)|highlights|gist|in a nutshell|"
                         r"what('s| is| are) (this|the|that|these|those|my)?\s*[\w\- ]{0,40}\s*(about|all about)|what does (this|the|that|my) [\w\- ]{0,30}(say|cover|contain|discuss)\s*\??$)\b", re.I)
_RE_PROJECT = re.compile(r"\b(what (does|is) (my|this|the|our) (project|repo|repository|codebase|app|application|program|code|tool|system|software)( \w+)?( do| about| for)?|"
                         r"explain (my|this|the|our) (project|repo|repository|codebase|app|application|program|code|software)|how (does|do) (my|this|the|our) (project|app|application|program|code|system|software|codebase) work|"
                         r"describe (my|this|the|our) (project|repo|repository|codebase|app|application|program|software))\b", re.I)
_RE_PROJECT2 = re.compile(r"\b(analy[sz]e|explain|describe|review|walk me through|break down|trace)\b.{0,50}\b(project|codebase|repo|repository|code|app|application|pipeline|system|software|implementation|model)\b", re.I)
_RE_CODE = re.compile(r"\b(how (is|are|does|do) .{0,60}(implemented|work|handled|done)|where is .{0,50}(implemented|defined|handled)|explain (the )?(function|class|method|module|script|code)|what does (the )?(function|class|method|module|script) .{0,40} do|"
                      r"which (function|class|method|module)|call(s|ed)? by|(function|class|method) (that|which))\b", re.I)
_RE_CALC = re.compile(r"\b(totals?|sums?|add up|averages?|avg|mean|median|minimum|maximum|min|max|highest|lowest|largest|smallest|biggest|top \d+|bottom \d+|count|how many (rows|records|entries|items|orders|customers|employees|people)|"
                      r"percent(age)?|share of|proportion|ratio|growth|grew|increase[ds]?|decrease[ds]?|change[ds]?|trends?|rate of|per (month|week|year|day|region|category)|by (month|week|year|day|region|category|quarter)|"
                      r"calculate|compute|how much (did|do|does|was|were|is|are)|standard deviation|variance|correlat\w+|breakdown|distribution)\b", re.I)
_RE_MULTI = re.compile(r"\b(across|all (the )?(documents|files|reports|notes|papers|sources)|each (document|file|report)|every (document|file|report)|overall|combined?|together|in total across|"
                       r"which (of the )?(documents|files|reports)|do (any|all) of|between (the )?(documents|files|reports)|consistent|conflict|agree|disagree|contradict)\b", re.I)
_RE_FINDFILE = re.compile(r"^\s*(please\s+)?(can you |could you )?(find|locate|search for|look for|show me|get me|list|recommend|suggest|pull up|which|where is|where are|where can i find)\b.{0,80}\b(files?|documents?|docs?|folders?|spreadsheets?|datasets?|csv|pdfs?|presentations?|decks?|scripts?|source code|notes?)\b", re.I)
_RE_FINDFILE2 = re.compile(r"^\s*(please\s+)?(find|locate|show me|get me|recommend|suggest|pull up)\b", re.I)
_RE_ASKS_ABOUT_CONTENT = re.compile(r"\b(say|says|said|state|states|mention|mentions|contain|contains|include|includes|according|stated|written|written in|info(rmation)? (on|about))\b", re.I)

_NUM_Q = re.compile(r"\b(how (many|much|long|often|big|large|high|low|old|far)|what('s| is| was| are| were)? (the )?(total|amount|number|count|percentage|percent|rate|price|cost|salary|budget|revenue|average|size|length|score|value|quantity|duration|limit|fee|deadline days)|percent|%|how many)\b", re.I)
_DATE_Q = re.compile(r"\b(when|what (date|day|year|month|time)|which (year|month|quarter|date)|deadline|due date|by when|how long ago|launch date|start date|end date)\b", re.I)
_PERSON_Q = re.compile(r"^\s*(who|whom|whose)\b|\bwho (is|was|are|were|will|did|does|leads?|manages?|owns?|wrote|authored)\b", re.I)
_WHY_Q = re.compile(r"^\s*why\b|\bwhat (is|was|were) the (reason|cause)\b", re.I)


@dataclass
class Need:
    text: str
    terms: List[str]               # surface terms (lowercase) that evidence must speak to
    stems: List[str]
    answer_type: str = "any"       # number | date | person | reason | any
    query: str = ""                # lexical search string (terms + synonyms)


@dataclass
class QuestionPlan:
    question: str
    kind: str                      # fact|summary|compare|data_calc|project_explain|code_explain|multi_doc|file_search|resume_improve
    needs: List[Need] = field(default_factory=list)
    target_file_ids: List[int] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    explicit_target: bool = False
    inherited: bool = False        # target taken from the previous turn
    type_group: Optional[str] = None
    find_plan: Optional[dict] = None   # legacy intent plan for file_search / resume
    confidence: float = 0.8
    notes: List[str] = field(default_factory=list)
    all_terms: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------- file mention detection
_NAMES_CACHE: Dict[str, tuple] = {}


def _owner_files(owner_id: str) -> List[dict]:
    hit = _NAMES_CACHE.get(owner_id)
    if hit and time.time() - hit[0] < 20:
        return hit[1]
    with get_cursor() as cur:
        cur.execute("""SELECT f.file_id, f.filename, f.file_path, f.file_type, s.state, s.meta_json, s.duplicate_of
                       FROM files f JOIN file_ingestion_state s ON s.file_id = f.file_id
                       JOIN connected_folders c ON c.folder_id = s.folder_id WHERE f.owner_id = %s;""", (owner_id,))
        rows = [dict(r) for r in cur.fetchall()]
    _NAMES_CACHE[owner_id] = (time.time(), rows)
    return rows


def _name_tokens(name: str) -> List[str]:
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name)
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", stem) if t]


_GENERIC_NAMES = {"notes", "note", "report", "reports", "summary", "document", "documents", "draft", "data", "final", "readme", "file", "files", "misc", "test", "tests",
                  "untitled", "new", "copy", "old", "backup", "temp", "todo", "list", "index", "main", "overview", "plan", "budget", "sales", "info"}


def _flat(s: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


def detect_file_mentions(query: str, owner_id: str) -> List[dict]:
    """A file is 'named' only when the question contains its full file name, or its name as one contiguous
    phrase ("the annual customer report" -> annual_customer_report.pdf). Topic words that merely happen to
    appear in a file name ("project alpha ... budget") do NOT make the question about that file."""
    q_raw = query.lower()
    q_flat = _flat(query)
    found = []
    for r in _owner_files(owner_id):
        name = r["filename"].lower()
        score = 0.0
        if name in q_raw:
            score = 3.0
        else:
            stem_flat = _flat(re.sub(r"\.[A-Za-z0-9]{1,5}$", "", r["filename"]))
            toks = stem_flat.split()
            if not toks:
                continue
            if stem_flat in q_flat and (len(toks) >= 2 or (len(toks[0]) >= 6 and toks[0] not in _GENERIC_NAMES)):
                score = 1.5 + 0.5 * len(toks)
        if score:
            found.append((score, r))
    if not found:
        return []
    found.sort(key=lambda x: -x[0])
    top = found[0][0]
    out, seen_names = [], set()
    for sc, r in found:
        if sc >= top - 0.6 and r["filename"] not in seen_names:
            out.append(r)
            seen_names.add(r["filename"])
    return out[:4]


# ---------------------------------------------------------------------------- planning
def _answer_type(q: str) -> str:
    if _PERSON_Q.search(q):
        return "person"
    if _DATE_Q.search(q):
        return "date"
    if _NUM_Q.search(q):
        return "number"
    if _WHY_Q.search(q):
        return "reason"
    return "any"


def _key_terms(text: str, drop: set) -> List[str]:
    out, seen = [], set()
    for t in T.content_terms(text):
        if t in FLUFF or t in drop:
            continue
        s = T._stem(t)
        if s not in seen:
            seen.add(s)
            out.append(t)
    return out


def _split_needs(question: str) -> List[str]:
    """Split a compound question into independent information needs ("X, and who Y?")."""
    q = question.strip().rstrip("?.! ")
    parts = re.split(r"\s*(?:;|\?\s+|,?\s+and\s+(?:also\s+)?(?=(?:what|who|when|where|why|how|which|is|are|does|do)\b)|,\s+(?=(?:what|who|when|where|why|how|which)\b))", q, flags=re.I)
    parts = [p.strip() for p in parts if p and len(T.content_terms(p)) >= 1]
    return parts if len(parts) > 1 else [q]


def _proper_terms(q: str) -> List[str]:
    """Stems of capitalised words that are not the sentence start or a question word ("Project Alpha")."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", q)
    out = []
    for i, w in enumerate(words):
        if i == 0 or not w[0].isupper() or w.lower() in T.STOP or w.lower() in ("what", "who", "when", "where", "why", "how", "which"):
            continue
        s = T._stem(w)
        if s not in out:
            out.append(s)
    return out


def _make_need(text: str, drop: set, inherit_terms: List[str]) -> Need:
    terms = _key_terms(text, drop)
    if not terms and inherit_terms:
        terms = list(inherit_terms)
    stems = [T._stem(t) for t in terms]
    expanded = []
    for t, s in zip(terms, stems):
        expanded.append(t)
        expanded.extend(x for x in synonym_stems(s) if x != s)
    return Need(text=text, terms=terms, stems=stems, answer_type=_answer_type(text), query=" ".join(dict.fromkeys(expanded)))


def plan_question(question: str, owner_id: str, conv: Optional[dict] = None) -> QuestionPlan:
    q = question.strip()
    conv = conv or {}
    ql = q.lower()
    notes: List[str] = []

    # --- which files is this about?
    mentions = detect_file_mentions(q, owner_id)
    target_ids = [m["file_id"] for m in mentions]
    target_names = [m["filename"] for m in mentions]
    explicit = bool(mentions)

    # --- follow-up resolution
    inherited = False
    focus = conv.get("focus_file_ids") or []
    words = T.tokens(q)
    is_follow = bool(focus) and not explicit and (
        bool(_FOLLOWUP.search(q)) or bool(_FOLLOWUP_START.search(q)))
    if is_follow and not _RE_PROJECT.search(q):
        target_ids, target_names, inherited = list(focus), list(conv.get("focus_names") or []), True
        notes.append("Follow-up: continuing with " + ", ".join(target_names[:3]))

    drop = set()
    for n in target_names:
        drop.update(_name_tokens(n))
    drop = {d for d in drop if d not in T.STOP} if explicit else set()

    inherit_terms = list(conv.get("topic_terms") or []) if (inherited and len(T.content_terms(q)) < 2) else []

    # --- question kind
    kind = "fact"
    find_plan = None
    from_resume = bool(re.search(r"\b(resume|résumé|cv|curriculum vitae)\b", ql))
    legacy = classify_find(q)
    asks_mechanism = bool(re.match(r"\s*(how|why|where|when|what (does|do|is|are))\b", q, re.I)) and not re.search(r"\b(improve|better|suggest|feedback|fix|enhance|strengthen|tips)\b", q, re.I)
    if from_resume and legacy["intent"] == "resume_improve" and not asks_mechanism:
        kind, find_plan = "resume_improve", legacy
    elif _RE_PROJECT.search(q) or _RE_PROJECT2.search(q):
        kind = "project_explain"
    elif _RE_COMPARE.search(q):
        kind = "compare"
    elif (_RE_FINDFILE.search(q) or (_RE_FINDFILE2.search(q) and not _RE_ASKS_ABOUT_CONTENT.search(q) and not _RE_CALC.search(q) and not _RE_SUMMARY.search(q))) and not inherited:
        kind, find_plan = "file_search", legacy
    elif _RE_SUMMARY.search(q) and len(_key_terms(q, drop)) <= 5 + len(drop):
        kind = "summary"
    elif _RE_CALC.search(q):
        kind = "data_calc"
    elif _RE_CODE.search(q) or legacy["intent"] == "find_code" and _WH.search(q):
        kind = "code_explain"
    elif _RE_MULTI.search(q):
        kind = "multi_doc"
    elif legacy["intent"] in ("find_dataset", "project_files") and not _WH.search(q):
        kind, find_plan = "file_search", legacy
    elif not (_WH.search(q) or q.endswith("?")) and not _RE_ASKS_ABOUT_CONTENT.search(q) and not _RE_SUMMARY.search(q):
        kind, find_plan = "file_search", legacy          # an imperative topic phrase such as "budget documents"

    # --- information needs
    needs: List[Need] = []
    if kind in ("fact", "multi_doc", "code_explain", "data_calc"):
        parts = _split_needs(q)
        subject = _proper_terms(q)
        for pi, part in enumerate(parts):
            nd = _make_need(part, drop, inherit_terms)
            # "who manages Project Alpha, and what is ITS budget?": the later part inherits the named subject
            if pi > 0 and subject and len(nd.terms) <= 1 and (re.search(r"\b(its|their|his|her|it|that|this)\b", part, re.I) or not set(T._stem(x) for x in nd.terms) & set(subject)):
                extra = [t for t in subject if t not in nd.stems]
                nd = Need(text=nd.text, terms=nd.terms + extra, stems=nd.stems + extra, answer_type=nd.answer_type, query=(nd.query + " " + " ".join(extra)).strip())
            needs.append(nd)
    elif kind in ("summary", "compare", "project_explain"):
        needs.append(_make_need(q, drop, inherit_terms))
    all_terms = list(dict.fromkeys(t for n in needs for t in n.terms))

    confidence = 0.9 if explicit or kind in ("file_search", "resume_improve") else 0.75
    if kind == "fact" and not all_terms:
        confidence = 0.3
        notes.append("No concrete topic words found")

    group = None
    if kind == "data_calc":
        group = "data"
    elif kind == "code_explain":
        group = "code"
    return QuestionPlan(question=q, kind=kind, needs=needs, target_file_ids=target_ids, target_names=target_names, explicit_target=explicit,
                        inherited=inherited, type_group=group, find_plan=find_plan, confidence=confidence, notes=notes, all_terms=all_terms)
