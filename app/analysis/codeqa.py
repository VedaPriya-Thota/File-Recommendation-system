"""
Reading source code WITHOUT running it. Python is parsed with `ast`; other languages use conservative regexes.
Nothing here imports, executes or evaluates user code. Everything reported is structure that is literally in the
file: definitions, docstrings, imports, calls, entry points. Behaviour beyond that is inference from reading,
and the assistant says so.
"""
import ast
import re
from typing import Dict, List, Optional

_DEF_RX = {
    "js": re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\s*\*?\s*([A-Za-z_$][\w$]*)|class\s+([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(?[^=]*?\)?\s*=>)", re.M),
    "java": re.compile(r"^\s*(?:public|private|protected|static|final|abstract|\s)*(?:class|interface|enum)\s+(\w+)|^\s*(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\], ]+\s+(\w+)\s*\(", re.M),
    "go": re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(|^type\s+(\w+)\s+(?:struct|interface)", re.M),
    "rs": re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)|^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", re.M),
    "generic": re.compile(r"^\s*(?:def|function|func|fn|sub)\s+(\w+)|^\s*class\s+(\w+)", re.M),
}
_IMPORT_RX = re.compile(r"^\s*(?:import\s+[\w.*{}, ]+(?:\s+from\s+['\"][^'\"]+['\"])?|from\s+[\w.]+\s+import\s+[\w*, ]+|const\s+\w+\s*=\s*require\(['\"][^'\"]+['\"]\)|use\s+[\w:]+|#include\s*[<\"][^>\"]+[>\"])", re.M)


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def outline_python(text: str) -> Optional[dict]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}      # functions only (not exception classes)
    items = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        calls = []
        call_nodes = sorted((n for n in ast.walk(node) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))   # source order
        for n in call_nodes:
            f = n.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name and name not in calls:
                calls.append(name)
        args = [a.arg for a in node.args.args] if not isinstance(node, ast.ClassDef) else []
        items.append({"kind": "class" if isinstance(node, ast.ClassDef) else "function", "name": node.name, "line_start": node.lineno,
                      "line_end": getattr(node, "end_lineno", node.lineno), "doc": (ast.get_docstring(node) or "").strip().split("\n\n")[0][:300],
                      "args": args, "calls": calls[:12], "calls_local": [c for c in calls if c in defined][:8],
                      "returns": any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node))})
    imports = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imports.append((n.module or "") + (" (relative)" if n.level else ""))
    has_main = any(isinstance(n, ast.If) and "__name__" in ast.dump(n.test) for n in tree.body)
    items.sort(key=lambda x: x["line_start"])
    return {"language": "Python", "doc": (ast.get_docstring(tree) or "").strip().split("\n\n")[0][:400], "items": items, "imports": sorted(set(imports))[:25], "entry_point": has_main}


def outline_generic(text: str, ext: str) -> dict:
    rx = _DEF_RX["js"] if ext in ("js", "jsx", "ts", "tsx") else _DEF_RX["java"] if ext in ("java", "kt", "cs") else _DEF_RX["go"] if ext == "go" else _DEF_RX["rs"] if ext == "rs" else _DEF_RX["generic"]
    items = []
    for m in rx.finditer(text):
        name = next((g for g in m.groups() if g), None)
        if name and name not in ("if", "for", "while", "switch", "catch"):
            line = _line_of(text, m.start())
            seg = text[m.start():m.start() + 160]
            items.append({"kind": "class" if re.match(r"\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:class|struct|interface|enum|type)\b", seg) else "function",
                          "name": name, "line_start": line, "line_end": line, "doc": _leading_comment(text, m.start()), "args": [], "calls": [], "calls_local": [], "returns": False})
    imports = [m.group(0).strip()[:100] for m in _IMPORT_RX.finditer(text)][:25]
    return {"language": None, "doc": "", "items": items[:80], "imports": imports, "entry_point": bool(re.search(r"\bmain\s*\(|if __name__|app\.listen|createServer", text))}


def _leading_comment(text: str, pos: int) -> str:
    before = text[:pos].rstrip().split("\n")
    out = []
    for line in reversed(before[-4:]):
        l = line.strip()
        if l.startswith(("//", "#", "*", "/*", "///")):
            out.append(l.lstrip("/#* ").strip())
        else:
            break
    return " ".join(reversed(out))[:300]


def outline(text: str, ext: str) -> dict:
    o = outline_python(text) if ext == "py" else None
    return o or outline_generic(text, ext)


_NOISE_CALLS = {"len", "str", "int", "float", "list", "dict", "set", "tuple", "sorted", "round", "range", "enumerate", "zip", "map", "filter", "min", "max", "sum", "any", "all",
                "print", "isinstance", "getattr", "setattr", "hasattr", "open", "super", "append", "extend", "update", "get", "items", "keys", "values", "join", "split",
                "strip", "format", "lower", "upper", "startswith", "endswith", "replace", "add", "pop", "copy", "step", "execute", "fetchone", "fetchall", "cursor"}


def describe_item(it: dict) -> str:
    """One factual sentence about a function/class built only from its signature, docstring and calls."""
    names = [c for c in (it.get("calls_local") or []) if c not in _NOISE_CALLS]
    if not names:
        names = [c for c in (it.get("calls") or []) if c not in _NOISE_CALLS and not c[:1].isupper() and len(c) > 3]
    if it["kind"] == "class":
        base = f"`{it['name']}` is a class (lines {it['line_start']}–{it['line_end']})"
    else:
        sig = f"({', '.join(a for a in it.get('args', []) if a != 'self')})" if it.get("args") else "()"
        base = f"`{it['name']}{sig}` (lines {it['line_start']}–{it['line_end']})"
    parts = []
    if it.get("doc"):
        parts.append(f"its docstring says “{_one_line(it['doc'])}”")
    if names:
        parts.append("it calls " + ", ".join(f"`{c}`" for c in list(dict.fromkeys(names))[:4]))
    if it.get("kind") == "function" and not parts:
        return base + " matches the question, but has no docstring or notable calls."
    return base + ": " + "; ".join(parts) + "."


def _one_line(s: str, n: int = 220) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def project_overview(files: List[dict]) -> dict:
    """files: [{filename, rel_path, file_type, text}] -> structured facts about a code project."""
    readme = next((f for f in files if re.match(r"readme(\.\w+)?$", f["filename"], re.I)), None)
    manifests = [f for f in files if f["filename"].lower() in ("package.json", "pyproject.toml", "requirements.txt", "setup.py", "cargo.toml", "go.mod", "pom.xml", "build.gradle")]
    code = [f for f in files if f["file_type"] in ("py", "js", "ts", "tsx", "jsx", "java", "go", "rs", "cs", "kt", "rb", "php", "c", "cpp", "swift")]
    parts = []
    for f in code[:60]:
        o = outline(f["text"], f["file_type"])
        parts.append({"filename": f["filename"], "rel_path": f["rel_path"], "file_id": f["file_id"], "outline": o, "lines": f["text"].count("\n") + 1})
    return {"readme": readme, "manifests": manifests, "modules": parts}


_IMPERATIVE = {"open", "run", "install", "clone", "pip", "npm", "cd", "python", "use", "then", "go", "download", "unzip", "set", "export", "start", "see", "note", "make", "create", "build", "visit"}


def readme_blurb(text: str, limit: int = 600) -> str:
    """The README's own description of the project: the first prose paragraph that is not a command, list,
    badge or install step. Empty string if the README has none (the caller then says so)."""
    paras = []
    for block in re.split(r"\n\s*\n", text):
        b = block.strip()
        if "\n" in b and b.startswith("#"):
            b = b.split("\n", 1)[1].strip()
        if not b or b.startswith(("#", "```", "![", "[![", "|", "---", "===", "*", "-", ">", "<", "$")) or re.match(r"^\d+[.)]\s", b) or re.match(r"^[\W_]+$", b):
            continue
        first = re.split(r"\W+", b.lower(), 1)[0]
        if first in _IMPERATIVE or "```" in b or b.count("`") > 6:
            continue
        if len(re.findall(r"[A-Za-z]{3,}", b)) < 8:
            continue
        paras.append(re.sub(r"\s+", " ", b))
        if sum(len(p) for p in paras) > limit:
            break
    return " ".join(paras)[:limit * 2]


# ----------------------------------------------------------------------------- cross-file workflow tracing
_CALL_ORDER_NOISE = _NOISE_CALLS - {"step", "get"}


def trace_workflow(files: List[dict], keywords: List[str], max_depth: int = 3, max_steps: int = 14) -> dict:
    """Follow how control flows through a set of code files, starting from the functions that match `keywords`.

    files: [{file_id, filename, rel_path, file_type, text}]. Pure static analysis (ast / regex): the code is never
    imported or executed. Returns {entry:[...], chain:[{depth, name, file, line_start, line_end, doc, calls}], imports, notes}.
    """
    defs: Dict[str, List[dict]] = {}
    per_file = {}
    for f in files:
        o = outline(f["text"], f["file_type"])
        lines = f["text"].split("\n")
        per_file[f["file_id"]] = o
        for it in o["items"]:
            if it["kind"] != "function":
                continue
            body = "\n".join(lines[it["line_start"] - 1: it["line_end"]])
            rec = {**it, "file_id": f["file_id"], "filename": f["filename"], "rel_path": f["rel_path"], "body": body}
            defs.setdefault(it["name"], []).append(rec)
    kw = [k.lower() for k in keywords if k]

    def score(d):
        hay = (d["name"] + " " + d["doc"]).lower()
        body = d["body"].lower()
        return 2.0 * sum(1 for k in kw if k in hay) + 0.5 * sum(1 for k in kw if k in body)

    all_defs = [d for ds in defs.values() for d in ds]
    seeds = sorted((d for d in all_defs if score(d) > 0), key=lambda d: (-score(d), d["rel_path"], d["line_start"]))[:6]
    called_by = {}
    for d in all_defs:
        for c in d["calls"]:
            if c in defs:
                called_by.setdefault(c, set()).add(d["name"])
    # entry points: matching functions nobody else in the set calls, or `main`/`train`-like drivers
    entries = [d for d in seeds if d["name"] not in called_by] or seeds[:1]
    chain, seen = [], set()

    def walk(d, depth):
        key = (d["file_id"], d["name"])
        if key in seen or len(chain) >= max_steps:
            return
        seen.add(key)
        local = [c for c in d["calls"] if c in defs and c not in _CALL_ORDER_NOISE]
        chain.append({"depth": depth, "name": d["name"], "file_id": d["file_id"], "file": d["filename"], "rel_path": d["rel_path"], "line_start": d["line_start"],
                      "line_end": d["line_end"], "doc": d["doc"], "args": d.get("args", []), "calls_here": local[:6],
                      "library_calls": [c for c in d["calls"] if c not in defs and c not in _CALL_ORDER_NOISE and not c[:1].isupper()][:8]})
        if depth < max_depth:
            for c in local[:6]:
                nxt = sorted(defs[c], key=lambda x: (x["file_id"] != d["file_id"], x["rel_path"]))[0]
                walk(nxt, depth + 1)

    for e in entries[:2]:
        walk(e, 0)
    imports = sorted({i for o in per_file.values() for i in o["imports"]})
    notes = []
    if not seeds:
        notes.append("no function matched the topic words")
    return {"entry": [e["name"] for e in entries[:2]], "chain": chain, "imports": imports[:30], "notes": notes, "seeds": [s["name"] for s in seeds]}


def read_config(text: str, ext: str) -> Dict[str, str]:
    """Flat key/value pairs from a JSON / YAML / TOML / INI / .env style config (no evaluation)."""
    out: Dict[str, str] = {}
    if ext == "json":
        import json
        try:
            obj = json.loads(text)
        except ValueError:
            return out

        def flat(o, pre=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    flat(v, f"{pre}{k}.")
            elif isinstance(o, list) and len(o) <= 6 and all(not isinstance(x, (dict, list)) for x in o):
                out[pre.rstrip(".")] = str(o)
            elif not isinstance(o, list):
                out[pre.rstrip(".")] = str(o)
        flat(obj)
    else:
        section = ""
        for line in text.split("\n"):
            l = line.strip()
            if not l or l.startswith(("#", ";", "//")):
                continue
            m = re.match(r"^\[([^\]]+)\]$", l)
            if m:
                section = m.group(1) + "."
                continue
            m = re.match(r"^([A-Za-z_][\w.\-]*)\s*[:=]\s*(.*?)\s*(?:#.*)?$", l)
            if m and m.group(2):
                out[section + m.group(1)] = m.group(2).strip("\"'")[:80]
    return dict(list(out.items())[:80])
