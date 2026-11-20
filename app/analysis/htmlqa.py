"""Structured reading of an HTML page (stdlib parser only; scripts are read as text and never executed).

extract_structure(html) -> {title, description, lang, headings[], nav_links[], forms[], buttons[], inputs[], images, links, scripts{external[], inline_functions[],
events[]}, stylesheets[], sections[], text_head, tables, iframes}. `describe(struct)` turns it into plain-English facts that are each
backed by something literally present in the markup.
"""
import re
from html.parser import HTMLParser
from typing import Dict, List


class _P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.s = {"title": "", "description": "", "lang": "", "headings": [], "nav_links": [], "forms": [], "buttons": [], "inputs": [], "images": 0, "links": 0,
                  "scripts": {"external": [], "inline_chars": 0}, "stylesheets": [], "sections": [], "tables": 0, "iframes": [], "canvas": 0, "media": 0}
        self._stack: List[str] = []
        self._text: List[str] = []
        self._cur = None
        self._form = None
        self._in = {"title": False, "script": False, "style": False, "nav": 0, "button": False, "a": None, "head": None}
        self.inline_js: List[str] = []
        self.visible: List[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        s = self.s
        if tag == "html":
            s["lang"] = a.get("lang", "")
        elif tag == "title":
            self._in["title"] = True
        elif tag == "meta" and a.get("name", "").lower() == "description":
            s["description"] = a.get("content", "")[:300]
        elif tag == "link" and "stylesheet" in a.get("rel", "").lower():
            s["stylesheets"].append(a.get("href", "")[:100])
        elif tag == "script":
            if a.get("src"):
                s["scripts"]["external"].append(a["src"][:120])
            self._in["script"] = True
        elif tag == "style":
            self._in["style"] = True
        elif tag == "nav":
            self._in["nav"] += 1
        elif tag in ("h1", "h2", "h3"):
            self._in["head"] = tag
            self._text = []
        elif tag == "a":
            s["links"] += 1
            self._in["a"] = {"href": a.get("href", "")[:100], "text": ""}
        elif tag == "form":
            self._form = {"action": a.get("action", ""), "method": (a.get("method") or "get").lower(), "id": a.get("id", ""), "fields": [], "submit": ""}
            s["forms"].append(self._form)
        elif tag in ("input", "textarea", "select"):
            f = {"tag": tag, "type": a.get("type", "text" if tag == "input" else tag), "name": a.get("name") or a.get("id", ""), "placeholder": a.get("placeholder", ""), "label": a.get("aria-label", "")}
            if f["type"] in ("submit", "button"):
                (self._form["submit"] if self._form is not None else None)
                if self._form is not None and not self._form["submit"]:
                    self._form["submit"] = a.get("value", "") or "submit"
                s["buttons"].append(a.get("value", "") or f["type"])
            elif f["type"] != "hidden":
                s["inputs"].append(f)
                if self._form is not None:
                    self._form["fields"].append(f["name"] or f["type"])
        elif tag == "button":
            self._in["button"] = True
            self._text = []
        elif tag == "img":
            s["images"] += 1
        elif tag == "table":
            s["tables"] += 1
        elif tag == "iframe":
            s["iframes"].append(a.get("src", "")[:100])
        elif tag == "canvas":
            s["canvas"] += 1
        elif tag in ("video", "audio"):
            s["media"] += 1
        if tag in ("section", "main", "header", "footer", "article", "aside") and (a.get("id") or a.get("class")):
            s["sections"].append(f"{tag}#{a['id']}" if a.get("id") else f"{tag}.{a['class'].split()[0]}")
        for k, v in a.items():
            if k.startswith("on") and v:
                self.inline_js.append(f"{k}: {v}")

    def handle_endtag(self, tag):
        s = self.s
        if tag == "title":
            self._in["title"] = False
        elif tag == "script":
            self._in["script"] = False
        elif tag == "style":
            self._in["style"] = False
        elif tag == "nav" and self._in["nav"]:
            self._in["nav"] -= 1
        elif tag in ("h1", "h2", "h3") and self._in["head"] == tag:
            txt = " ".join("".join(self._text).split())
            if txt:
                s["headings"].append({"level": int(tag[1]), "text": txt[:120]})
            self._in["head"] = None
        elif tag == "a" and self._in["a"] is not None:
            if self._in["nav"]:
                s["nav_links"].append({"text": " ".join(self._in["a"]["text"].split())[:50], "href": self._in["a"]["href"]})
            self._in["a"] = None
        elif tag == "button" and self._in["button"]:
            txt = " ".join("".join(self._text).split())
            s["buttons"].append(txt[:40] or "button")
            if self._form is not None and not self._form["submit"]:
                self._form["submit"] = txt[:40]
            self._in["button"] = False
        elif tag == "form":
            self._form = None

    def handle_data(self, data):
        if self._in["title"]:
            self.s["title"] += data
        elif self._in["script"]:
            self.inline_js.append(data)
            self.s["scripts"]["inline_chars"] += len(data)
        elif self._in["style"]:
            pass
        else:
            if self._in["head"] or self._in["button"]:
                self._text.append(data)
            if self._in["a"] is not None:
                self._in["a"]["text"] += data
            t = data.strip()
            if t:
                self.visible.append(t)


def extract_structure(html: str) -> Dict:
    p = _P()
    try:
        p.feed(html)
        p.close()
    except Exception:  # noqa: BLE001 -- malformed markup: report what was parsed
        pass
    s = p.s
    s["title"] = " ".join(s["title"].split())[:150]
    js = "\n".join(p.inline_js)
    funcs = re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>", js)
    s["scripts"]["inline_functions"] = list(dict.fromkeys(a or b for a, b in funcs))[:20]
    events = re.findall(r"(?:getElementById|querySelector)\(['\"]([^'\"]+)['\"]\)\s*\.addEventListener\(['\"](\w+)['\"]|addEventListener\(['\"](\w+)['\"]", js)
    s["scripts"]["events"] = list(dict.fromkeys((f"{ev or ev2} on {sel}" if sel else (ev or ev2)) for sel, ev, ev2 in events))[:12]
    s["scripts"]["uses_fetch"] = bool(re.search(r"\bfetch\(|XMLHttpRequest|axios", js))
    s["scripts"]["uses_storage"] = bool(re.search(r"localStorage|sessionStorage", js))
    s["scripts"]["inline_handlers"] = [x for x in p.inline_js if re.match(r"on\w+:", x)][:8]
    s["text_head"] = " ".join(p.visible)[:500]
    s["words_visible"] = len(" ".join(p.visible).split())
    return s


def describe(st: Dict) -> List[Dict]:
    """Facts about the page, each tagged with how it was established (explicit markup vs. inference)."""
    facts = []

    def add(text, kind="explicit", needle=None):
        facts.append({"text": text, "kind": kind, "needle": needle})
    if st["title"]:
        add(f"The page is titled “{st['title']}”.", needle=st["title"])
    if st["description"]:
        add(f"Its meta description says: “{st['description']}”.", needle=st["description"][:60])
    h1 = [h["text"] for h in st["headings"] if h["level"] == 1]
    if h1:
        add(f"The main heading is “{h1[0]}”.", needle=h1[0])
    subs = [h["text"] for h in st["headings"] if h["level"] >= 2][:6]
    if subs:
        add("Its sections include " + ", ".join(f"“{x}”" for x in subs) + ".", needle=subs[0])
    if st["nav_links"]:
        add(f"There is a navigation menu with {len(st['nav_links'])} link(s): " + ", ".join(x["text"] or x["href"] for x in st["nav_links"][:6]) + ".", needle=(st["nav_links"][0]["text"] or st["nav_links"][0]["href"]))
    for f in st["forms"][:3]:
        fields = ", ".join(x for x in f["fields"] if x)[:80]
        add(f"It contains a form{(' (' + f['id'] + ')') if f['id'] else ''} with fields {fields or '(no named fields)'}"
            + (f" and a “{f['submit']}” button" if f["submit"] else "") + (f", which submits to {f['action']}" if f["action"] else "") + ".", needle=(f["id"] or f["action"] or "<form"))
    if st["buttons"] and not st["forms"]:
        add("Buttons: " + ", ".join(f"“{b}”" for b in dict.fromkeys(st["buttons"]) if b)[:200] + ".")
    sc = st["scripts"]
    if sc["external"]:
        add("It loads external scripts: " + ", ".join(x.split("/")[-1] or x for x in sc["external"][:4]) + ".", needle=sc["external"][0])
    if sc["inline_functions"]:
        add("Its own JavaScript defines " + ", ".join(f"`{f}()`" for f in sc["inline_functions"][:6]) + ".", needle=sc["inline_functions"][0])
    if sc["events"] or sc["inline_handlers"]:
        add("It responds to user actions (" + ", ".join((sc["events"] + [h.split(":")[0] for h in sc["inline_handlers"]])[:5]) + ").")
    if sc["uses_fetch"]:
        add("The script makes network requests (fetch/XHR), so the page loads or sends data dynamically.")
    if sc["uses_storage"]:
        add("It stores data in the browser (localStorage/sessionStorage).")
    if st["canvas"]:
        add(f"It draws on {st['canvas']} canvas element(s), which suggests graphics, charts or a game.", "inferred")
    if st["tables"]:
        add(f"It displays {st['tables']} table(s) of data.")
    if st["images"]:
        add(f"It shows {st['images']} image(s).")
    # purpose: inferred only from the evidence above
    kinds = []
    if st["forms"] and any(x["type"] == "password" for x in st["inputs"]):
        kinds.append("a login or sign-up page")
    elif st["forms"]:
        kinds.append("a page that collects user input")
    if st["canvas"] or sc["events"]:
        kinds.append("an interactive application")
    if st["tables"]:
        kinds.append("a data display")
    if not kinds and st["words_visible"] > 80:
        kinds.append("a content page (mostly text)")
    if not kinds:
        kinds.append("a simple static page")
    joined = kinds[0] if len(kinds) == 1 else ", ".join(kinds[:-1]) + " and " + kinds[-1]
    add("Taken together, this looks like " + joined + ".", "inferred")
    return facts
