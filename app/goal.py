"""
Deterministic understand_goal -- architecture_final.md §4, §Phase1 prompt.

No LLM. A practical heuristic: keyword-based file_type inference, a small
relative-date parser, stopword-filtered topic token extraction, and a
filename-candidate regex. Ambiguity is a deterministic function of how many
strong signals were found -- when the heuristic isn't confident, the
uncertainty is preserved in `ambiguity_score` rather than inventing filters.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

# --- file type inference ------------------------------------------------
FILE_TYPE_KEYWORDS = {
    "pptx": ["slide deck", "slides", "presentation", "deck", "pptx", "powerpoint"],
    "xlsx": ["spreadsheet", "budget sheet", "excel", "xlsx", "workbook"],
    "docx": ["document", "doc", "notes", "docx", "word doc", "write-up", "writeup"],
    "pdf": ["pdf"],
    "png": ["photo", "image", "picture", "png", "screenshot"],
}

# --- stopwords for topic-token extraction --------------------------------
# NOTE: subject-matter words like "budget" must stay as TOPIC tokens (they
# are the actual thing being searched for); only filler/date/pronoun words
# go here. File-type-signal words (e.g. "slide", "deck") are filtered
# separately below via FILE_TYPE_SIGNAL_WORDS so they don't double-count.
STOPWORDS = {
    "a", "an", "the", "i", "me", "my", "was", "were", "is", "are", "am",
    "find", "get", "show", "search", "for", "of", "about", "on", "in",
    "at", "to", "from", "that", "this", "it", "its", "thing", "things",
    "looking", "look", "edited", "edit", "opened", "open", "modified",
    "last", "week", "weeks", "month", "months", "day", "days", "yesterday",
    "today", "ago", "recently", "recent", "file", "files", "document",
    "documents", "and", "or", "with", "up", "please", "can", "you",
}
# File-type-signal words are excluded from topic_tokens (they already
# became `filters.file_type`), so they don't double count as "topic".
FILE_TYPE_SIGNAL_WORDS = {
    "slide", "deck", "slides", "presentation", "powerpoint",
    "spreadsheet", "excel", "workbook",
    "document", "doc", "word",
    "pdf", "photo", "image", "picture", "png", "screenshot",
}

FILENAME_HINT_RE = re.compile(
    r"(?:called|named|titled)\s+['\"]?([\w\-. ]+?)['\"]?(?:[.,]|$)", re.IGNORECASE
)
EXPLICIT_FILE_RE = re.compile(r"\b[\w\-]+\.(pptx|xlsx|docx|pdf|png|jpg|jpeg|csv)\b", re.IGNORECASE)


@dataclass
class Goal:
    intent: str
    filters: dict
    topic_tokens: List[str]
    concept_text: str
    ambiguity_score: float
    raw_query: str

    def to_dict(self):
        return {
            "intent": self.intent,
            "filters": self.filters,
            "topic_tokens": self.topic_tokens,
            "concept_text": self.concept_text,
            "ambiguity_score": self.ambiguity_score,
            "raw_query": self.raw_query,
        }


def _keyword_pattern(kw: str) -> "re.Pattern":
    """Whole-token match (Phase 11): the keyword delimited by non-alphanumerics (so `_`, `.`, `-` and spaces
    are separators, as in filenames), optionally pluralised ("documents", "slides", "pdfs"). Plain substring
    matching misread "doctor" as `doc`, "photosynthesis" as `photo`, "excellent" as `excel`, "landslides" as
    `slides` and "representations" as `presentation`, and the resulting file_type filter then silently removed
    every relevant file from bm25/semantic as well."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(kw)}(?:e?s)?(?![a-z0-9])")


_FILE_TYPE_PATTERNS = {ft: [_keyword_pattern(kw) for kw in kws] for ft, kws in FILE_TYPE_KEYWORDS.items()}


def _infer_file_type(query_lower: str) -> Optional[str]:
    for file_type, patterns in _FILE_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(query_lower):
                return file_type
    return None


def _infer_date_range(query_lower: str, now: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
    if "today" in query_lower:
        return now - timedelta(days=1), now
    if "yesterday" in query_lower:
        return now - timedelta(days=2), now - timedelta(hours=12)
    if "last week" in query_lower or "past week" in query_lower:
        return now - timedelta(days=9), now
    if "this week" in query_lower:
        return now - timedelta(days=7), now
    if "last month" in query_lower or "past month" in query_lower:
        return now - timedelta(days=35), now
    if "this month" in query_lower:
        return now - timedelta(days=30), now
    return None, None


def _infer_filename_candidate(raw_query: str) -> Optional[str]:
    m = EXPLICIT_FILE_RE.search(raw_query)
    if m:
        return m.group(0)
    m = FILENAME_HINT_RE.search(raw_query)
    if m:
        return m.group(1).strip()
    return None


def _extract_topic_tokens(query_lower: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", query_lower)
    tokens = []
    for w in words:
        if w in STOPWORDS or w in FILE_TYPE_SIGNAL_WORDS:
            continue
        if len(w) <= 1:
            continue
        tokens.append(w)
    # de-duplicate, preserve order
    seen = set()
    ordered = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _compute_ambiguity(
    file_type: Optional[str],
    filename_candidate: Optional[str],
    date_range: Tuple,
    topic_tokens: List[str],
) -> float:
    """Deterministic ambiguity heuristic: start from 'fully ambiguous' (1.0)
    and subtract confidence for each strong signal found. Clamped to
    [0.0, 1.0]. This is a config item tunable via evaluation
    (architecture_final.md §19), not a fixed constant in spirit -- but for
    Phase 1 the weights themselves are hard-coded and simple.
    """
    score = 1.0
    if filename_candidate:
        score -= 0.45
    if file_type:
        score -= 0.25
    if date_range[0] is not None:
        score -= 0.15
    if topic_tokens:
        # more topic tokens -> more confidence, up to a cap
        score -= min(0.25, 0.1 * len(topic_tokens))
    return round(max(0.0, min(1.0, score)), 4)


def understand_goal(raw_query: str, now: Optional[datetime] = None) -> Goal:
    now = now or datetime.now(timezone.utc)
    query_lower = raw_query.lower()

    file_type = _infer_file_type(query_lower)
    date_from, date_to = _infer_date_range(query_lower, now)
    filename_candidate = _infer_filename_candidate(raw_query)
    topic_tokens = _extract_topic_tokens(query_lower)

    ambiguity_score = _compute_ambiguity(
        file_type, filename_candidate, (date_from, date_to), topic_tokens
    )

    return Goal(
        intent="file_discovery",
        filters={
            "file_type": file_type,
            "date_range": {"from": date_from, "to": date_to} if date_from else None,
            "filename_candidate": filename_candidate,
        },
        topic_tokens=topic_tokens,
        concept_text=" ".join(topic_tokens) if topic_tokens else raw_query,
        ambiguity_score=ambiguity_score,
        raw_query=raw_query,
    )
