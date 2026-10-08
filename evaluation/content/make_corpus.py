"""Regenerates evaluation/content/corpus/ from corpus_spec.py. Provenance only: the generated files are
committed and the evaluation reads them; reportlab is needed only to run THIS script."""
import os
import shutil
import textwrap

from evaluation.content.corpus_spec import CORPUS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def _write_pdf(path, text):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(path, pagesize=A4, invariant=1)
    y = 800
    for line in textwrap.wrap(text, 90):
        c.drawString(60, y, line)
        y -= 18
    c.save()


def _write_docx(path, text):
    from docx import Document
    d = Document()
    for para in text.split("\n"):
        d.add_paragraph(para)
    d.save(path)


def main():
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)
    for owner, files in CORPUS.items():
        os.makedirs(os.path.join(ROOT, owner))
        for name, text in files.items():
            path = os.path.join(ROOT, owner, name)
            ext = name.rsplit(".", 1)[1]
            if ext == "pdf":
                _write_pdf(path, text)
            elif ext == "docx":
                _write_docx(path, text)
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text + "\n")


if __name__ == "__main__":
    main()
