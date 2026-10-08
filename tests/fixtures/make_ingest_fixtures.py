"""Regenerates the two binary fixtures (PDF, DOCX) from fixed text. Provenance only --
the generated files are committed, tests never run this."""
import os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingest_corpus")
PDF_LINES = [
    "Garden planting guide.",
    "Tomato seedlings need rich soil, full sun and a regular watering schedule.",
    "Mulch the beds in spring to keep weeds down and the roots cool.",
]
DOCX_PARAS = [
    "Hiring plan.",
    "We will recruit three engineers this quarter and run an onboarding programme for new employees.",
]


def main():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(os.path.join(HERE, "sample_document.pdf"), pagesize=A4, invariant=1)
    y = 800
    for line in PDF_LINES:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    from docx import Document
    d = Document()
    for p in DOCX_PARAS:
        d.add_paragraph(p)
    d.save(os.path.join(HERE, "sample_document.docx"))


if __name__ == "__main__":
    main()
