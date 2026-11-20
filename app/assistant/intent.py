"""Deterministic request understanding for the assistant (no model call): which kind of job is this, what
should be searched for, and which file families are relevant."""
import re
from typing import Dict, List

from app.ingestion.extract import TYPE_GROUPS

INTENTS = ("resume_improve", "compare", "summarize", "analyze_data", "find_dataset", "find_code", "project_files", "recommend")

_STRIP = [
    r"\b(which|what)\s+(documents?|files?|notes?|reports?|spreadsheets?|pdfs?|presentations?)\s+(mention|contain|include|discuss|talk about|cover|reference|have|are about|relate to)s?\b",
    r"\b(mention|mentions|mentioning|contain|contains|containing|discuss|discusses|reference|references)\b",
    r"\b(and|then)\s+(also\s+)?(suggest|give|provide|propose|recommend|offer|tell)\b.*$",
    r"\b(please|can you|could you|would you|help me( to)?|i want to|i need to|i'd like to|i would like to|let's)\b",
    r"\b(find|locate|search( for)?|look( for| up)?|show( me)?|get( me)?|retrieve|pull up|identify|list|recommend|discover)\b",
    r"\b(trends?|insights?|statistics|stats|summari[sz]e|summary of|analy[sz]e|analysis of|compare|comparison (of|between)|review|critique|improve|difference(s)? between)\b",
    r"\b(all|every|any)\s+(of\s+)?(the\s+|my\s+)?(files?|documents?|folders?)\s+(that\s+(are|is)\s+|which\s+(are|is)\s+)?(related to|relating to|about|for|on|regarding|concerning|connected to|associated with|belonging to)\b",
    r"\b(files?|documents?)\s+(related to|about|for|on|regarding|concerning)\b",
    r"\b(datasets?|data sets?)\s+(relevant to|related to|about|for|on|regarding|concerning)\b",
    r"\b(source code|code|implementation|script|function|class)\s+(that|which)\s+(implements?|handles?|does|provides?|performs?)\b",
    r"\b(that|which)\s+(implements?|handles?)\b",
    r"\b(my|the|a|an|some|those|these|this|our)\b",
    r"[?!.]+",
]
_STRIP_RX = [re.compile(p, re.I) for p in _STRIP]

_SPLIT_COMPARE = re.compile(r"\s+(?:and|vs\.?|versus|with|against|to|or)\s+", re.I)


def _has(q: str, *pats) -> bool:
    return any(re.search(p, q) for p in pats)


def classify(query: str) -> Dict:
    q = " " + query.lower().strip() + " "
    data_word = r"\b(csv|tsv|excel|xlsx|spreadsheet|worksheet|dataset|data ?set|data|table|figures|numbers|sales|revenue|metrics)\b"
    analyze_verb = r"\b(analy[sz]e|analysis|trends?|statistics|stats|insights?|correlat\w*|average|growth|forecast|patterns?|outliers?|distribution|summari[sz]e the data)\b"
    if _has(q, r"\b(resume|résumé|cv|curriculum vitae)\b"):
        intent = "resume_improve" if _has(q, r"\b(improv\w*|suggest\w*|feedback|review|critique|better|fix|enhance|strengthen|tips|polish|edit)\b") else "recommend"
        group = None
        if intent == "recommend":
            intent = "recommend"
        return _pack(query, "resume_improve" if intent == "resume_improve" else "recommend", group, resume=True)
    if _has(q, r"\b(compare|comparison|versus|vs\.?|differences? between|how does .* differ|contrast)\b"):
        return _pack(query, "compare", None)
    if _has(q, analyze_verb) and _has(q, data_word):
        return _pack(query, "analyze_data", "data")
    if _has(q, r"\b(summari[sz]e|summary|tl;?dr|key points|main points|gist|overview of)\b"):
        return _pack(query, "summarize", None)
    if _has(q, r"\b(datasets?|data ?sets?)\b") or (_has(q, r"\b(csv|xlsx|excel|spreadsheet|json data)\b") and _has(q, r"\b(find|search|look|show|get|any|relevant)\b")):
        return _pack(query, "find_dataset", "data")
    if _has(q, r"\b(source code|codebase|implement(s|ation|ed|ing)?|function|class|script|module|endpoint|algorithm|repository|repo|library|api handler|unit tests?|python|javascript|typescript|java|golang|rust|c\+\+)\b"):
        return _pack(query, "find_code", "code")
    if _has(q, r"\b(project|related to|all files|everything (about|on|for))\b"):
        return _pack(query, "project_files", None)
    return _pack(query, "recommend", None)


def _pack(query: str, intent: str, group, resume: bool = False) -> Dict:
    cleaned = clean_query(query)
    if group:  # file-family words describe the container, not the topic
        slim = re.sub(r"\b(csv|tsv|excel|xlsx|spreadsheets?|worksheets?|datasets?|data ?sets?|data|json|files?|code|source|script)\b", " ", cleaned, flags=re.I)
        slim = re.sub(r"\s+(and|or|of|in)\s*$|^\s*(and|or|of|in)\s+", " ", re.sub(r"\s+", " ", slim)).strip()
        cleaned = slim or cleaned
    if resume:
        cleaned = (cleaned + " resume cv curriculum vitae").strip() if not re.search(r"resume|cv", cleaned, re.I) else cleaned + " cv curriculum vitae"
    parts = []
    if intent == "compare":
        pieces = [p.strip() for p in _SPLIT_COMPARE.split(clean_query(query)) if p.strip()]
        if len(pieces) >= 2:
            parts = pieces[:3]
    types = TYPE_GROUPS.get(group, []) if group else []
    return {"intent": intent, "retrieval_query": cleaned or query.strip(), "type_group": group, "types": types, "compare_parts": parts,
            "max_results": 15 if intent in ("project_files",) else 8}


def clean_query(query: str) -> str:
    s = query
    for rx in _STRIP_RX:
        s = rx.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" ,;:-")
