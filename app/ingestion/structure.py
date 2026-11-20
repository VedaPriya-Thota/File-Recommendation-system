"""
Document structure: deterministic, pure functions that turn a file's extracted text (+ extraction metadata) into
PASSAGES that keep related sentences together and know where they came from (page / slide / sheet, section
heading, line range). The same passages are used for the embedding index (one Qdrant point per passage) and for
the assistant's passage-level BM25, so a hit always maps back to an exact, verifiable location.

Passage text is always the verbatim slice `text[start:end]`.
"""
import bisect
import re
from dataclasses import dataclass
from typing import List, Optional

STRUCTURE_VERSION = 2          # bump to make already-indexed files re-chunk on the next scan
TARGET_CHARS = 900             # ~130-150 words: fits embedding models and keeps a topic together
MAX_CHARS = 1500
MIN_CHARS = 200                # smaller trailing/leading fragments are merged into a neighbour

_HEADING_NUM = re.compile(r"^(\d+(\.\d+){0,3}[.)]?|[IVXLC]+\.|[A-Z]\.)\s+\S")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_CODE_BOUNDARY = re.compile(r"^(?:async\s+def |def |class |function |export |public |private |protected |static |func |fn |impl |interface |struct |module\.exports)", re.M)


@dataclass
class Passage:
    idx: int
    start: int
    end: int
    page: Optional[int] = None        # page / slide number (1-based) when the format has them
    page_label: Optional[str] = None  # "Page" | "Slide" | "Sheet"
    sheet: Optional[str] = None
    heading: Optional[str] = None
    line_start: int = 1
    line_end: int = 1

    def location(self) -> str:
        bits = []
        if self.page is not None:
            bits.append(f"{self.page_label or 'Page'} {self.page}")
        if self.sheet:
            bits.append(f"sheet “{self.sheet}”")
        if self.heading:
            bits.append(f"section “{self.heading[:70]}”")
        if not bits or self.page_label == "Line":
            bits.append(f"lines {self.line_start}–{self.line_end}")
        return ", ".join(bits)

    def to_payload(self) -> dict:
        return {"start": self.start, "end": self.end, "page": self.page, "heading": self.heading,
                "line_start": self.line_start, "line_end": self.line_end, "sheet": self.sheet, "page_label": self.page_label}


def is_heading(line: str) -> bool:
    l = line.strip()
    if not l or len(l) > 90 or l.endswith((",", ";")):
        return False
    if l.startswith("#"):
        return True
    if _HEADING_NUM.match(l) and len(l.split()) <= 12 and not l.endswith("."):
        return True
    letters = [c for c in l if c.isalpha()]
    if len(letters) >= 4 and l.isupper() and len(l.split()) <= 10:
        return True
    return False


_SMALL = {"of", "and", "the", "for", "to", "in", "on", "a", "an", "with", "by", "at", "or", "vs", "from"}


def is_title_line(line: str) -> bool:
    """Short Title-Case line without sentence punctuation: 'Findings', 'Methodology', 'Risks and Mitigations'."""
    l = line.strip()
    if not l or len(l) > 55 or re.search(r"[.!?:;,]$", l) or "|" in l:
        return False
    words = l.split()
    if not (1 <= len(words) <= 6):
        return False
    if sum(c.isdigit() for c in l) > 4:
        return False
    return all(w[0].isupper() or w.lower() in _SMALL or w[0].isdigit() for w in words) and l[0].isupper()


def clean_heading(line: str) -> str:
    return line.strip().lstrip("#").strip().rstrip(":")


def _line_starts(text: str) -> List[int]:
    starts = [0]
    for m in re.finditer(r"\n", text):
        starts.append(m.end())
    return starts


def _split_long(text: str, s: int, e: int) -> List[tuple]:
    """Split an oversized block at sentence boundaries into <= TARGET-ish slices."""
    out, cur = [], s
    for m in _SENT_END.finditer(text, s, e):
        if m.start() - cur >= TARGET_CHARS:
            out.append((cur, m.start()))
            cur = m.end()
    if cur < e:
        out.append((cur, e))
    # hard cut for text without sentence punctuation (logs, minified text)
    final = []
    for a, b in out:
        while b - a > MAX_CHARS:
            cut = text.rfind(" ", a, a + MAX_CHARS)
            cut = cut if cut > a + MAX_CHARS // 2 else a + MAX_CHARS
            final.append((a, cut))
            a = cut
        final.append((a, b))
    return final


def build_passages(text: str, meta: Optional[dict] = None, file_type: str = "") -> List[Passage]:
    meta = meta or {}
    if not text or not text.strip():
        return []
    code = bool(meta.get("language")) and file_type not in ("html", "css", "xml", "yaml", "yml", "toml", "ini", "cfg")
    lines_at = _line_starts(text)
    page_offsets = meta.get("page_offsets") or []
    label = meta.get("page_label") or "Page"
    sheets = meta.get("sheet_offsets") or []   # [(start, name)]

    # 1. blocks: contiguous runs separated by blank lines; code additionally breaks at top-level definitions
    blocks = []   # (start, end, heading_line_text or None)
    pos = 0
    for m in re.finditer(r"\n\s*\n", text):
        blocks.append((pos, m.start()))
        pos = m.end()
    blocks.append((pos, len(text)))
    blocks = [(a, b) for a, b in blocks if text[a:b].strip()]
    if not code:
        split_blocks = []
        for a, b in blocks:                       # headings inside a block (typical for PDF text): start a new block there
            cuts, lines = [a], text[a:b].split("\n")
            off = a
            prev = ""
            for li, line in enumerate(lines):
                if li > 0 and prev.rstrip().endswith((".", "!", "?")) and li + 1 < len(lines) and (is_heading(line) or is_title_line(line)):
                    cuts.append(off)
                prev = line
                off += len(line) + 1
            cuts.append(b)
            for x, y in zip(cuts, cuts[1:]):
                if text[x:y].strip():
                    split_blocks.append((x, y))
        blocks = split_blocks
    if code:
        refined = []
        for a, b in blocks:
            cuts = [a] + [a + m.start() for m in _CODE_BOUNDARY.finditer(text[a:b]) if m.start() > 0 and text[a + m.start() - 1] == "\n"] + [b]
            for x, y in zip(cuts, cuts[1:]):
                if text[x:y].strip():
                    refined.append((x, y))
        blocks = refined

    # 2. group blocks into passages; headings start a new passage and stay attached to what follows
    groups: List[list] = []   # each: [start, end, heading]
    cur_heading: Optional[str] = None
    for a, b in blocks:
        first_line = text[a:b].split("\n", 1)[0]
        has_body = "\n" in text[a:b].strip()
        title_like = (not code) and is_title_line(first_line) and (has_body or len(text[a:b].strip()) < 60)
        heading_block = (not code) and (is_heading(first_line) or title_like) and len(text[a:b].strip()) <= 140
        if heading_block:
            cur_heading = clean_heading(first_line)
            groups.append([a, b, cur_heading, True])   # True = pure heading (merge forward)
            continue
        if (not code) and (is_heading(first_line) or title_like):
            cur_heading = clean_heading(first_line)
            groups.append([a, b, cur_heading, False])
            continue
        if groups and groups[-1][3]:                    # attach to a preceding pure heading
            groups[-1][1], groups[-1][3] = b, False
            continue
        if groups and (groups[-1][1] - groups[-1][0]) + (b - a) <= TARGET_CHARS and not _boundary(groups[-1], a, page_offsets, sheets, code):
            groups[-1][1] = b
            continue
        groups.append([a, b, cur_heading, False])

    # 3. split oversized groups, merge tiny fragments, then materialise
    spans = []
    for a, b, h, _ in groups:
        if b - a > MAX_CHARS:
            for x, y in _split_long(text, a, b):
                spans.append([x, y, h])
        else:
            spans.append([a, b, h])
    merged: List[list] = []
    for sp in spans:
        if merged and (sp[1] - sp[0] < MIN_CHARS) and merged[-1][2] == sp[2] and (sp[1] - merged[-1][0]) <= MAX_CHARS \
                and not _boundary([merged[-1][0], merged[-1][1]], sp[0], page_offsets, sheets, code):
            merged[-1][1] = sp[1]
        else:
            merged.append(sp)

    out: List[Passage] = []
    for i, (a, b, h) in enumerate(merged):
        p = Passage(idx=i, start=a, end=b, heading=h)
        if page_offsets:
            p.page = max(1, bisect.bisect_right(page_offsets, a))
            p.page_label = label
        if sheets:
            j = bisect.bisect_right([s for s, _ in sheets], a) - 1
            if j >= 0:
                p.sheet = sheets[j][1]
        p.line_start = bisect.bisect_right(lines_at, a)
        p.line_end = bisect.bisect_right(lines_at, max(a, b - 1))
        if code:
            p.page_label = "Line"
        out.append(p)
    return out


def _boundary(group, next_start, page_offsets, sheets, code) -> bool:
    """True if joining text at next_start to `group` would cross a page/slide/sheet boundary."""
    a = group[0]
    if page_offsets and bisect.bisect_right(page_offsets, a) != bisect.bisect_right(page_offsets, next_start):
        return True
    if sheets:
        keys = [s for s, _ in sheets]
        if bisect.bisect_right(keys, a) != bisect.bisect_right(keys, next_start):
            return True
    return False
