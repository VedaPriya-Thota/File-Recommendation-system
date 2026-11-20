"""Rule-based résumé review over the real extracted text. Every issue quotes the actual line it found, so the
advice is checkable. (An optional LLM may rewrite/expand it; these findings are always computed first.)"""
import re
from typing import Dict, List

from app.analysis.text import tokens

SECTIONS = {
    "summary": r"^(professional )?(summary|profile|objective|about me)\b",
    "experience": r"^(work |professional |relevant )?(experience|employment|work history)\b",
    "education": r"^education\b|^academic",
    "skills": r"^(technical |core |key )?skills\b|^competenc|^technologies",
    "projects": r"^(selected |key |personal )?projects\b",
    "certifications": r"^(certifications?|licenses?|awards|achievements|publications)\b",
}
WEAK = ["responsible for", "duties included", "worked on", "helped", "assisted", "various", "etc", "in charge of", "tried to",
        "familiar with", "hard-working", "team player", "detail-oriented", "results-driven", "go-getter", "references available"]
STRONG_VERBS = set("""led built designed developed launched increased reduced improved delivered created managed implemented
optimized automated achieved drove owned established migrated architected analyzed negotiated mentored shipped scaled generated
streamlined resolved coordinated spearheaded engineered deployed""".split())
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")
LINK = re.compile(r"(linkedin\.com/\S+|github\.com/\S+)", re.I)
BULLET = re.compile(r"^\s*([-•*▪●◦·]|\d+[.)])\s+(.*)")
METRIC = re.compile(r"\d+(\.\d+)?\s?(%|x\b|k\b|m\b|million|billion|users|customers|ms|hours|days|\$)|[$€£]\s?\d|\b\d{2,}\b")


def review(text: str, pages: int = None) -> Dict:
    lines = [l.rstrip() for l in text.split("\n")]
    nonblank = [l for l in lines if l.strip()]
    words = len(tokens(text))
    checks: List[Dict] = []

    def add(area, ok, title, detail, evidence=None, weight=1):
        checks.append({"area": area, "ok": ok, "title": title, "detail": detail, "evidence": evidence or [], "weight": weight})

    # contact
    email, phone, link = EMAIL.search(text), PHONE.search(text), LINK.search(text)
    missing = [n for n, v in (("email", email), ("phone number", phone), ("LinkedIn/GitHub link", link)) if not v]
    add("Contact", not missing[:2] or len(missing) <= 1, "Contact details",
        "Found: " + ", ".join(n for n, v in (("email", email), ("phone", phone), ("profile link", link)) if v) + ("." if not missing else f". Missing: {', '.join(missing)}."), weight=2)

    # sections
    present = {}
    for ln in nonblank:
        l = ln.strip().lstrip("#").strip().rstrip(":")
        if len(l) > 40:
            continue
        for sec, rx in SECTIONS.items():
            if re.search(rx, l, re.I) and sec not in present:
                present[sec] = l
    needed = ["experience", "education", "skills"]
    miss = [s for s in needed if s not in present]
    add("Structure", not miss, "Core sections", "Sections detected: " + (", ".join(present) or "none") +
        (f". Add a clear heading for: {', '.join(miss)}." if miss else "."), weight=3)
    if "summary" not in present:
        add("Structure", False, "No summary/profile", "A 2–3 line summary at the top helps recruiters grasp your focus in seconds.", weight=1)

    # length
    if pages:
        ok = pages <= 2
        add("Length", ok, "Length", f"{pages} page(s), ~{words} words." + ("" if ok else " Aim for 1–2 pages (1 page under ~10 years' experience)."), weight=2)
    else:
        ok = 250 <= words <= 900
        add("Length", ok, "Length", f"~{words} words." + ("" if ok else (" That is quite short—add concrete achievements." if words < 250 else " That is long—trim to the most relevant 1–2 pages.")), weight=2)

    # bullets / impact
    bullets = [m.group(2).strip() for l in nonblank if (m := BULLET.match(l))]
    if not bullets:
        bullets = [l.strip() for l in nonblank if 40 < len(l.strip()) < 220 and l.strip()[0].isupper()]
    quant = [b for b in bullets if METRIC.search(b)]
    ratio = len(quant) / len(bullets) if bullets else 0
    unq = [b for b in bullets if not METRIC.search(b)][:3]
    add("Impact", ratio >= 0.4, "Quantified achievements",
        f"{len(quant)} of {len(bullets)} bullet/achievement lines contain numbers or measurable outcomes ({ratio:.0%})." +
        ("" if ratio >= 0.4 else " Add metrics (%, $, time saved, users, scale) to more bullets."), unq, weight=3)

    # action verbs
    starts = [re.sub(r"[^a-z]", "", b.split()[0].lower()) for b in bullets if b.split()]
    strong = sum(1 for s in starts if s in STRONG_VERBS or s.endswith("ed") and len(s) > 4)
    weak_starts = [b for b in bullets if b.split() and b.split()[0].lower().rstrip(",") in ("responsible", "duties", "helped", "worked", "assisted", "was", "were", "involved")][:3]
    add("Language", not weak_starts and (not starts or strong / len(starts) >= 0.5), "Action verbs",
        f"{strong} of {len(starts)} bullets start with an action verb." + (" Rewrite passive openings with verbs like Led, Built, Reduced, Delivered." if weak_starts or (starts and strong / len(starts) < 0.5) else ""), weak_starts, weight=2)

    # weak phrases / cliches
    found = []
    for l in nonblank:
        for w in WEAK:
            if re.search(rf"\b{re.escape(w)}\b", l, re.I):
                found.append(l.strip()[:140])
                break
    add("Language", not found, "Vague or clichéd phrases", f"{len(found)} line(s) use filler such as 'responsible for', 'helped', 'team player'." if found else "No common filler phrases found.", found[:4], weight=1)

    # first person
    fp = [l.strip()[:120] for l in nonblank if re.search(r"\b(I|my|me)\b", l) and len(l) > 20][:3]
    add("Language", not fp, "First-person pronouns", "Résumés read stronger without 'I/my'." if fp else "No first-person pronouns.", fp, weight=1)

    # dates
    has_dates = len(re.findall(r"\b(19|20)\d{2}\b", text))
    add("Content", has_dates >= 2, "Dates", f"{has_dates} year reference(s) found." + ("" if has_dates >= 2 else " Add start/end dates to roles and education."), weight=1)

    # long paragraphs
    para = [l.strip()[:100] + "…" for l in nonblank if len(l) > 350][:2]
    add("Readability", not para, "Dense paragraphs", "Long unbroken paragraphs found; use short bullets." if para else "Lines are scannable.", para, weight=1)

    total = sum(c["weight"] for c in checks)
    got = sum(c["weight"] for c in checks if c["ok"])
    score = round(100 * got / total) if total else 0
    suggestions = [f"{c['title']}: {c['detail']}" for c in checks if not c["ok"]]
    return {"score": score, "checks": checks, "suggestions": suggestions, "stats": {"words": words, "bullets": len(bullets), "quantified_bullets": len(quant), "sections": list(present)}}
