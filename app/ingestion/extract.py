"""
Content extraction boundary. Pure: bytes in, ExtractedContent out; no filesystem, database or network.

Supported: .txt .md .rst .log .pdf .docx .pptx .xlsx/.xlsm .csv/.tsv .json/.jsonl and common source-code /
config files (see CODE_LANGUAGES). Legacy binary Office formats (.doc .xls .ppt) are reported "unsupported"
(metadata-only) rather than guessed at.

Each extraction also returns small, factual `metadata` (page/slide/sheet counts, columns, line counts ...)
that the UI and the analysis tools show. Failures are reported as a short `error_code` only -- never the
exception message, which could contain paths or document content.
"""
import csv
import io
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from app.ingestion.normalize import normalize_code, normalize_text

OK, EMPTY, UNSUPPORTED, FAILED = "ok", "empty", "unsupported", "failed"

# Upper bound for text kept per file (keeps the DB, BM25 and embedding cost bounded for huge files).
MAX_TEXT_CHARS = 400_000
CSV_TEXT_ROWS = 400          # rows rendered into searchable text for tabular files (stats use the real file)
SHEET_TEXT_ROWS = 300

CODE_LANGUAGES: Dict[str, str] = {
    "py": "Python", "js": "JavaScript", "jsx": "JavaScript (JSX)", "ts": "TypeScript", "tsx": "TypeScript (TSX)",
    "java": "Java", "kt": "Kotlin", "c": "C", "h": "C header", "cpp": "C++", "cc": "C++", "hpp": "C++ header",
    "cs": "C#", "go": "Go", "rs": "Rust", "rb": "Ruby", "php": "PHP", "swift": "Swift", "scala": "Scala",
    "r": "R", "m": "Objective-C/MATLAB", "sql": "SQL", "sh": "Shell", "bash": "Shell", "ps1": "PowerShell",
    "bat": "Batch", "lua": "Lua", "dart": "Dart", "html": "HTML", "css": "CSS", "scss": "SCSS", "vue": "Vue",
    "svelte": "Svelte", "yaml": "YAML", "yml": "YAML", "toml": "TOML", "ini": "INI", "cfg": "Config",
    "xml": "XML", "ipynb": "Jupyter Notebook", "tf": "Terraform", "proto": "Protobuf", "gradle": "Gradle",
}
TEXT_EXTENSIONS = ("txt", "md", "markdown", "rst", "log", "tex")
TABULAR = ("csv", "tsv")
SUPPORTED_EXTENSIONS = tuple(sorted(set(
    TEXT_EXTENSIONS + TABULAR + ("pdf", "docx", "pptx", "xlsx", "xlsm", "json", "jsonl") + tuple(CODE_LANGUAGES))))

# Type groups used by the assistant / UI filters.
TYPE_GROUPS = {
    "document": ["pdf", "docx", "txt", "md", "markdown", "rst", "tex", "pptx"],
    "data": ["csv", "tsv", "xlsx", "xlsm", "json", "jsonl"],
    "code": sorted(CODE_LANGUAGES),
}


@dataclass(frozen=True)
class ExtractedContent:
    status: str                       # ok | empty | unsupported | failed
    text: str = ""                    # NORMALIZED text (empty unless status == ok)
    error_code: Optional[str] = None  # set only when status == failed
    metadata: dict = field(default_factory=dict)


def _join_parts(parts):
    """parts: [(label, text)] -> (joined_text, offsets). Each part is normalized first so that normalizing the
    joined text again is the identity (offsets stay valid). offsets[i] = start of part i (empty parts point at
    the start of the next non-empty one)."""
    texts = [normalize_text(t) for _, t in parts]
    offsets, pos, chunks = [], 0, []
    for t in texts:
        offsets.append(pos)
        if t:
            chunks.append(t)
            pos += len(t) + 2
    return "\n\n".join(chunks), offsets


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            if enc == "utf-16" and not data[:2] in (b"\xff\xfe", b"\xfe\xff"):
                continue
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:4096] and data[:2] not in (b"\xff\xfe", b"\xfe\xff")


def _extract_plain(data: bytes) -> Tuple[str, dict]:
    text = _decode(data)
    return text, {"lines": text.count("\n") + 1, "words": len(text.split())}


def _extract_code(ext: str):
    def fn(data: bytes) -> Tuple[str, dict]:
        if _looks_binary(data):
            raise ValueError("binary")
        text = _decode(data)
        return text, {"language": CODE_LANGUAGES[ext], "lines": text.count("\n") + 1}
    return fn


def _extract_pdf(data: bytes) -> Tuple[str, dict]:
    from pypdf import PdfReader  # lazy: a missing optional library degrades to failed("extractor_unavailable")

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        if not reader.decrypt(""):
            raise PermissionError("encrypted_pdf")
    pages = [(page.extract_text() or "") for page in reader.pages]
    joined, offsets = _join_parts([("page", p) for p in pages])
    meta = {"pages": len(pages), "page_offsets": offsets, "page_label": "Page"}
    try:
        info = reader.metadata
        if info:
            for k, key in (("title", "title"), ("author", "author")):
                v = getattr(info, key, None)
                if v:
                    meta[k] = str(v)[:200]
    except Exception:  # noqa: BLE001
        pass
    return joined, meta


def _extract_docx(data: bytes) -> Tuple[str, dict]:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = []
    for p in doc.paragraphs:
        style = (getattr(getattr(p, "style", None), "name", "") or "").lower()
        if p.text.strip() and (style.startswith("heading") or style == "title"):
            parts.append("# " + p.text.strip())    # makes the section structure visible to the passage builder
        else:
            parts.append(p.text)
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    meta = {"paragraphs": len(doc.paragraphs), "tables": len(doc.tables)}
    cp = doc.core_properties
    for k in ("title", "author"):
        v = getattr(cp, k, None)
        if v:
            meta[k] = str(v)[:200]
    return "\n".join(parts), meta


def _extract_pptx(data: bytes) -> Tuple[str, dict]:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    slides, n = [], 0
    for n, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(c.text for c in row.cells))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text
            if notes.strip():
                parts.append("Notes: " + notes)
        slides.append(("slide", "\n".join(parts)))
    joined, offsets = _join_parts(slides)
    return joined, {"slides": n, "page_offsets": offsets, "page_label": "Slide"}


def _extract_xlsx(data: bytes) -> Tuple[str, dict]:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    parts, sheets = [], []
    try:
        for ws in wb.worksheets:
            rows = 0
            header = None
            lines = []
            for row in ws.iter_rows(values_only=True):
                if all(c is None or str(c).strip() == "" for c in row):
                    continue
                rows += 1
                if header is None:
                    header = [str(c) if c is not None else "" for c in row]
                if rows <= SHEET_TEXT_ROWS:
                    lines.append(" | ".join("" if c is None else str(c) for c in row))
            parts.append((ws.title, "\n".join(lines)))
            sheets.append({"name": ws.title, "rows": rows, "columns": [h for h in (header or []) if h][:40]})
    finally:
        wb.close()
    joined, offsets = _join_parts(parts)
    sheet_offsets = [(o, name) for o, (name, _) in zip(offsets, parts)]
    return joined, {"sheets": sheets, "sheet_count": len(sheets), "sheet_offsets": sheet_offsets}


def _extract_csv(ext: str):
    def fn(data: bytes) -> Tuple[str, dict]:
        if _looks_binary(data):
            raise ValueError("binary")
        text = _decode(data)
        sample = text[:20000]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel_tab if ext == "tsv" else csv.excel
        reader = csv.reader(io.StringIO(text), dialect)
        header, rows, kept = None, 0, []
        for row in reader:
            if not any(c.strip() for c in row):
                continue
            rows += 1
            if header is None:
                header = row
            if rows <= CSV_TEXT_ROWS:
                kept.append(" | ".join(row))
        return "\n".join(kept), {"rows": max(rows - 1, 0), "columns": [h.strip() for h in (header or [])][:80],
                                 "delimiter": getattr(dialect, "delimiter", ",")}
    return fn


def _extract_json(data: bytes) -> Tuple[str, dict]:
    if _looks_binary(data):
        raise ValueError("binary")
    text = _decode(data)
    meta: dict = {}
    try:
        obj = json.loads(text)
        meta["json_type"] = type(obj).__name__
        if isinstance(obj, list):
            meta["items"] = len(obj)
            if obj and isinstance(obj[0], dict):
                meta["keys"] = list(obj[0].keys())[:40]
        elif isinstance(obj, dict):
            meta["keys"] = list(obj.keys())[:40]
        text = json.dumps(obj, ensure_ascii=False, indent=1)  # one value per line -> good chunks
    except ValueError:
        meta["json_valid"] = False  # JSON Lines or damaged: index the raw text
        meta["items"] = text.count("\n") + 1
    return text, meta


_EXTRACTORS: Dict[str, Callable[[bytes], Tuple[str, dict]]] = {
    **{e: _extract_plain for e in TEXT_EXTENSIONS},
    **{e: _extract_code(e) for e in CODE_LANGUAGES},
    "pdf": _extract_pdf, "docx": _extract_docx, "pptx": _extract_pptx,
    "xlsx": _extract_xlsx, "xlsm": _extract_xlsx,
    "csv": _extract_csv("csv"), "tsv": _extract_csv("tsv"),
    "json": _extract_json, "jsonl": _extract_json,
}


class ContentExtractor:
    def supports(self, extension: str) -> bool:
        return (extension or "").lower() in _EXTRACTORS

    def extract(self, extension: str, data: bytes) -> ExtractedContent:
        ext = (extension or "").lower()
        fn = _EXTRACTORS.get(ext)
        if fn is None:
            return ExtractedContent(UNSUPPORTED)
        try:
            raw, meta = fn(data)
        except ImportError:
            return ExtractedContent(FAILED, error_code="extractor_unavailable")
        except PermissionError:
            return ExtractedContent(FAILED, error_code="encrypted_" + ext)
        except ValueError as exc:
            if str(exc) == "binary":
                return ExtractedContent(FAILED, error_code="binary_content")
            return ExtractedContent(FAILED, error_code=f"malformed_{ext}")
        except Exception:  # noqa: BLE001 -- any parser error on a bad file is a per-file failure, never a crash
            return ExtractedContent(FAILED, error_code=f"malformed_{ext}")
        text = normalize_code(raw) if ext in CODE_LANGUAGES else normalize_text(raw)
        if text != raw:                         # normalization moved characters: offsets would lie -> drop them
            meta.pop("page_offsets", None)
            meta.pop("sheet_offsets", None)
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
            meta["truncated"] = True
        from app.ingestion.structure import STRUCTURE_VERSION
        meta["_sv"] = STRUCTURE_VERSION
        return ExtractedContent(OK, text, metadata=meta) if text else ExtractedContent(EMPTY, metadata=meta)
