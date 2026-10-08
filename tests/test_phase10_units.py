"""
Phase 10 -- pure-unit tests for the ingestion building blocks: file source,
extraction, normalization. No database, no Qdrant, no network.
"""
import builtins
import os
import shutil
import sys
from pathlib import Path

import pytest

from app.ingestion import ContentExtractor, LocalDirectorySource, SourceUnavailableError, normalize_text
from app.ingestion.source import FileTooLargeError

FIXTURES = Path(__file__).parent / "fixtures" / "ingest_corpus"


@pytest.fixture
def corpus(tmp_path):
    dest = tmp_path / "corpus"
    shutil.copytree(FIXTURES, dest)
    return dest


# ---------------------------------------------------------------- discovery
def test_discovery_is_recursive_deterministic_and_includes_unsupported_files(corpus):
    found = LocalDirectorySource(corpus).discover()
    assert [f.rel_path for f in found] == [
        "empty.txt", "finance_report.txt", "notes/standup.txt", "retrieval_notes.md",
        "sample_document.docx", "sample_document.pdf", "unsupported.xyz"]
    assert found == LocalDirectorySource(corpus).discover()  # same order, same metadata, every time
    by_path = {f.rel_path: f for f in found}
    assert by_path["unsupported.xyz"].extension == "xyz"      # metadata-only file is still discovered
    assert by_path["notes/standup.txt"].filename == "standup.txt" and by_path["empty.txt"].size_bytes == 0
    assert all("\\" not in f.rel_path and not f.rel_path.startswith("/") for f in found)  # relative POSIX only


def test_discovery_never_reads_file_contents(corpus, monkeypatch):
    def no_open(*a, **k):
        raise AssertionError("discovery must not open files")
    monkeypatch.setattr(builtins, "open", no_open)
    assert len(LocalDirectorySource(corpus).discover()) == 7


def test_symlinks_are_skipped_and_reads_cannot_escape_the_root(corpus, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    os.symlink(outside, corpus / "link.txt")
    src = LocalDirectorySource(corpus)
    assert "link.txt" not in [f.rel_path for f in src.discover()]
    with pytest.raises(PermissionError):
        src.read_bytes("../outside.txt", 1000)
    assert src.read_bytes("finance_report.txt", 10_000).startswith(b"The company")
    with pytest.raises(FileTooLargeError):
        src.read_bytes("finance_report.txt", 10)


def test_missing_root_is_an_explicit_error_not_an_empty_listing(tmp_path):
    with pytest.raises(SourceUnavailableError):
        LocalDirectorySource(tmp_path / "does-not-exist").discover()


# --------------------------------------------------------------- extraction
@pytest.mark.parametrize("name,expected", [
    ("finance_report.txt", "earnings grew strongly"),
    ("retrieval_notes.md", "nearest neighbour"),
    ("sample_document.pdf", "Tomato seedlings"),
    ("sample_document.docx", "recruit three engineers"),
])
def test_supported_formats_extract_real_content(name, expected):
    r = ContentExtractor().extract(name.rsplit(".", 1)[1], (FIXTURES / name).read_bytes())
    assert r.status == "ok" and expected in r.text and r.error_code is None


def test_empty_whitespace_only_and_unsupported_and_failures_are_distinct_statuses():
    ex = ContentExtractor()
    assert ex.extract("txt", b"").status == "empty"
    assert ex.extract("md", b"  \n\t \r\n ").status == "empty"
    assert ex.extract("xyz", b"data").status == "unsupported"
    assert ex.extract("pptx", b"data").status == "unsupported"  # deliberately out of scope
    bad_pdf, bad_docx = ex.extract("pdf", b"%PDF-garbage"), ex.extract("docx", b"not a zip")
    assert (bad_pdf.status, bad_pdf.error_code) == ("failed", "malformed_pdf")
    assert (bad_docx.status, bad_docx.error_code) == ("failed", "malformed_docx")
    assert bad_pdf.text == "" and bad_docx.text == ""


def test_failure_reports_a_code_never_an_exception_message_or_content():
    r = ContentExtractor().extract("pdf", b"SECRET-CONTENT-123 not a pdf")
    assert r.error_code == "malformed_pdf" and "SECRET" not in repr(r)


def test_missing_optional_library_degrades_to_a_failed_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)  # makes `from pypdf import ...` raise ImportError
    r = ContentExtractor().extract("pdf", (FIXTURES / "sample_document.pdf").read_bytes())
    assert (r.status, r.error_code) == ("failed", "extractor_unavailable")


def test_non_utf8_and_binary_text_files_do_not_crash_or_leak_nul_bytes():
    r = ContentExtractor().extract("txt", b"caf\xe9 \x00 bytes \xff\xfe ok")
    assert r.status == "ok" and "\x00" not in r.text and "bytes" in r.text


# ------------------------------------------------------------ normalization
def test_normalization_whitespace_line_endings_and_blank_lines():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"
    assert normalize_text("a   b\t\tc \u00a0 d") == "a b c d"
    assert normalize_text("p1\n\n\n\n\np2") == "p1\n\np2"
    assert normalize_text("  line one  \n   line two   ") == "line one\nline two"
    assert normalize_text("a\x0cb\u2028c") == "a\nb\nc"  # form feed / line separator become newlines


def test_normalization_unicode_and_invisible_characters():
    assert normalize_text("\ufb01nancial \uff21\uff29") == "financial AI"      # ligature + full-width fold (NFKC)
    assert normalize_text("zero\u200bwidth\u00adhyphen\ufeff") == "zerowidthhyphen"
    assert normalize_text("nul\x00byte\x07bell") == "nulbytebell"


def test_normalization_empty_and_idempotent():
    for empty in (None, "", "   ", "\n\n\t\r\n", "\u200b\u00a0"):
        assert normalize_text(empty) == ""
    text = " Mixed \r\n\n\n content\t here \u00a0 "
    assert normalize_text(normalize_text(text)) == normalize_text(text)
