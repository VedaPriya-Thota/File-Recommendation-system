"""
Questions about spreadsheet data, answered by REAL computation (never by a model's arithmetic).

  parse   : which column(s) does the question mean? which operation (sum/average/min/max/count/share/growth/
            top-N/trend)? grouped by what? restricted to which values / year / month?
  compute : pure Python over the actual rows (csv / openpyxl)
  explain : a sentence, a result table and honest data-quality notes (missing / non-numeric cells, rows used)

If the question cannot be mapped to columns and an operation with confidence, `answer()` says so instead of
guessing.
"""
import math
import re
import statistics
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from app.analysis import tabular
from app.analysis import text as T

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
MONTH_ABBR = {k[:3]: v for k, v in MONTHS.items()}
COL_SYNONYMS = {
    "revenue": ["sales", "income", "turnover", "earnings"], "sales": ["revenue", "income", "turnover"], "cost": ["expense", "spend", "spending", "expenditure", "costs"],
    "expense": ["cost", "spend", "spending"], "spend": ["cost", "expense", "spending", "budget"], "budget": ["spend", "allocation"], "profit": ["margin", "net", "earnings"],
    "headcount": ["employees", "staff", "people", "workers"], "employees": ["headcount", "staff"], "price": ["cost", "amount", "rate"], "amount": ["total", "value", "price"],
    "date": ["day", "week", "month", "period", "time"], "quantity": ["qty", "units", "count", "volume"], "region": ["area", "territory", "country", "market"],
}


_SYN_STEMMED = {T._stem(k): [T._stem(x) for x in v] for k, v in COL_SYNONYMS.items()}


def _norm(s: str) -> List[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
    return [T._stem(t) for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


class Table:
    def __init__(self, name: str, rows: List[list]):
        self.name = name
        self.headers = tabular._unique_headers(rows[0]) if rows else []
        n = len(self.headers)
        self.rows = [r + [None] * (n - len(r)) if len(r) < n else r[:n] for r in rows[1:]]
        self.types: Dict[str, str] = {}
        self.num: Dict[str, List[Optional[float]]] = {}
        self.dates: Dict[str, List[Optional[datetime]]] = {}
        self._infer()

    def _infer(self):
        for ci, h in enumerate(self.headers):
            raw = [r[ci] for r in self.rows]
            present = [v for v in raw if v is not None and str(v).strip() != ""]
            if not present:
                self.types[h] = "empty"
                continue
            nums = [tabular._to_number(v) for v in raw]
            dts = [tabular._to_date(v) for v in raw]
            n_num = sum(1 for x in nums if x is not None)
            n_dt = sum(1 for x in dts if x is not None)
            if n_num >= 0.8 * len(present) and not (h.lower() in ("year", "yr") and False):
                self.types[h] = "numeric"
                self.num[h] = nums
            elif n_dt >= 0.8 * len(present):
                self.types[h] = "date"
                self.dates[h] = dts
            else:
                self.types[h] = "category"
        # a numeric column that looks like a year acts as a date axis too
        for h in list(self.num):
            vals = [x for x in self.num[h] if x is not None]
            if h.lower() in ("year", "yr") and vals and all(1900 <= v <= 2100 and float(v).is_integer() for v in vals):
                self.dates[h] = [datetime(int(x), 1, 1) if x is not None else None for x in self.num[h]]

    def col(self, h: str) -> List:
        return [r[self.headers.index(h)] for r in self.rows]


def load_tables(data: bytes, ext: str) -> List[Table]:
    sheets = tabular._read_xlsx(data) if ext in ("xlsx", "xlsm") else tabular._read_csv(data)
    return [Table(name, s["rows"]) for name, s in sheets.items() if len(s["rows"]) >= 2]


# ---------------------------------------------------------------------------- parsing
def _col_score(col: str, q_stems: set, q_text: str) -> float:
    toks = _norm(col)
    if not toks:
        return 0.0
    score = 0.0
    for t in toks:
        if t in q_stems:
            score += 1.0
        elif any(x in q_stems for x in _SYN_STEMMED.get(t, [])):
            score += 0.7
    score /= len(toks)
    if col.lower() in q_text.lower():
        score += 0.5
    return score


def _find_cols(table: Table, q_stems: set, q: str, kinds: Sequence[str], exclude=()) -> List[str]:
    scored = [(_col_score(h, q_stems, q), h) for h in table.headers if table.types.get(h) in kinds and h not in exclude]
    scored = [x for x in scored if x[0] >= 0.5]
    scored.sort(key=lambda x: -x[0])
    return [h for _, h in scored]


def _group_col(table: Table, q: str, q_stems: set, value_cols: Sequence[str]) -> Optional[str]:
    # "top 3 regions by revenue" / "which region had the highest sales": the NOUN names the grouping column
    nm = re.search(r"\btop \d+ (\w+)|\bwhich (\w+) (?:had|has|have|is|was|did|generated|made|brought|recorded|performed)\b|\bwhat (\w+) (?:had|has|have|is|was) the (?:highest|lowest|most|least|best|worst)|\bwhich (\w+) (?:has|had) the\b", q, re.I)
    if nm:
        noun = next(g for g in nm.groups() if g)
        if noun.lower() in ("month", "year", "quarter", "week", "day") and any(table.types.get(h) == "date" or h in table.dates for h in table.headers):
            return "__period:" + noun.lower()
        for h in table.headers:
            if h not in value_cols and table.types.get(h) == "category" and T._stem(noun) in _norm(h):
                return h
    m = re.search(r"\b(?:by|per|for each|each|across|grouped by|broken down by|split by|group by|in each)\s+(?:the\s+)?([A-Za-z_][\w\- ]{1,30}?)(?=\s+(?:and|in|for|during|from|where|with|$)|[?.,]|$)", q, re.I)
    if not m:
        return None
    words = _norm(m.group(1))
    best, bs = None, 0.0
    for h in table.headers:
        if h in value_cols or table.types.get(h) not in ("category", "date", "numeric"):
            continue
        toks = _norm(h)
        sc = len(set(toks) & set(words)) / max(len(toks), 1)
        if sc > bs:
            best, bs = h, sc
    if bs >= 0.5:
        return best
    # "by month/year/quarter" on a date column
    if words and words[0] in ("month", "year", "quarter", "week", "day"):
        return "__period:" + words[0]
    return None


def _filters(table: Table, q: str, skip_cols: Sequence[str]):
    """Category values mentioned in the question, plus a year / month restriction for the date column."""
    ql = q.lower()
    out = []
    for h in table.headers:
        if table.types.get(h) != "category" or h in skip_cols:
            continue
        vals = OrderedDict()
        for v in table.col(h):
            if v is not None and str(v).strip():
                vals.setdefault(str(v).strip(), 0)
        if len(vals) > 60:
            continue
        for v in vals:
            if len(v) >= 2 and re.search(rf"(?<![A-Za-z0-9]){re.escape(v.lower())}(?![A-Za-z0-9])", ql):
                out.append((h, v))
    years = sorted({int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", q)})
    months = [MONTHS[m] for m in MONTHS if re.search(rf"\b{m}\b", ql)] + [MONTH_ABBR[a] for a in MONTH_ABBR if re.search(rf"\b{a}\b(?!\w)", ql) and a not in ("mar", "may") or re.search(rf"\b{a}\.? \d{{4}}\b", ql)]
    months = sorted(set(months))
    quarters = [int(x) for x in re.findall(r"\bq([1-4])\b", ql)]
    return out, years, months, quarters


def _date_col(table: Table, q_stems: set, q: str) -> Optional[str]:
    if not table.dates:
        return None
    sc = sorted(((_col_score(h, q_stems, q), h) for h in table.dates), key=lambda x: -x[0])
    return sc[0][1]


def _row_mask(table: Table, cat_filters, years, months, quarters, date_col):
    keep = []
    for i, row in enumerate(table.rows):
        ok = True
        by_col = defaultdict(list)
        for h, v in cat_filters:
            by_col[h].append(v)
        for h, vals in by_col.items():       # several values of one column are alternatives; different columns combine with AND
            cell = row[table.headers.index(h)]
            if str(cell).strip() not in vals:
                ok = False
        if ok and date_col and (years or months or quarters):
            d = table.dates[date_col][i]
            if d is None or (years and d.year not in years) or (months and d.month not in months) or (quarters and (d.month - 1) // 3 + 1 not in quarters):
                ok = False
        if ok:
            keep.append(i)
    return keep


# ---------------------------------------------------------------------------- compute
def _agg(values: List[float], op: str):
    if not values:
        return None
    if op == "sum":
        return sum(values)
    if op == "avg":
        return statistics.fmean(values)
    if op == "median":
        return statistics.median(values)
    if op == "min":
        return min(values)
    if op == "max":
        return max(values)
    if op == "count":
        return len(values)
    if op == "std":
        return statistics.pstdev(values)
    return None


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and not x.is_integer():
        return f"{x:,.2f}"
    return f"{int(x):,}"


def _detect_op(q: str):
    ql = q.lower()
    if re.search(r"\b(growth|grew|grown|change[ds]?|increase[ds]?|decrease[ds]?|decline[ds]?|drop(ped)?|rise|rose|trends?)\b", ql):
        return "change"
    if re.search(r"\b(percent(age)?|share|proportion|%|fraction)\b", ql):
        return "share"
    if re.search(r"\b(median)\b", ql):
        return "median"
    if re.search(r"\b(standard deviation|std)\b", ql):
        return "std"
    if re.search(r"\b(average|avg|mean|typical)\b", ql):
        return "avg"
    if re.search(r"\b(highest|largest|biggest|maximum|max|most|top|best|peak)\b", ql):
        return "max"
    if re.search(r"\b(lowest|smallest|minimum|min|least|worst|bottom)\b", ql):
        return "min"
    if re.search(r"\b(how many|count|number of)\b", ql):
        return "count"
    if re.search(r"\b(total|sum|add up|altogether|combined|overall|how much)\b", ql):
        return "sum"
    return None


def answer(question: str, tables: List[Table], file_label: str) -> dict:
    """-> {ok, sentence, table:[[...]], operation, notes:[...], rows_used, column, sheet, source_rows:[[...]]}"""
    q = question.strip()
    q_stems = {T._stem(t) for t in T.tokens(q)}
    op = _detect_op(q)
    # pick the sheet whose columns match the question best
    best, best_sc = None, -1.0
    for t in tables:
        sc = sum(_col_score(h, q_stems, q) for h in t.headers)
        if sc > best_sc:
            best, best_sc = t, sc
    t = best
    if t is None:
        return {"ok": False, "reason": "no table"}
    notes: List[str] = []
    val_cols = _find_cols(t, q_stems, q, ("numeric",))
    group = _group_col(t, q, q_stems, val_cols)
    cat_filters, years, months, quarters = _filters(t, q, [group] if group else [])
    date_col = _date_col(t, q_stems, q)
    if (years or months or quarters) and not date_col:
        return {"ok": False, "reason": "The question restricts a period, but the table has no date column I can read.", "columns": t.headers}
    keep = _row_mask(t, cat_filters, years, months, quarters, date_col)
    if not keep:
        return {"ok": False, "reason": "No rows match those conditions (" + ", ".join(f"{h} = {v}" for h, v in cat_filters) + (f", year {years}" if years else "") + ").", "columns": t.headers}
    mnames = [list(MONTHS)[m - 1].capitalize() for m in months]
    period = " ".join(x for x in (", ".join(mnames), ("Q" + ", Q".join(map(str, quarters))) if quarters else "", ", ".join(map(str, years))) if x)
    cond = [f"{h} = {v}" for h, v in cat_filters] + ([period] if period else [])
    cond_txt = (" for " + " and ".join(cond)) if cond else ""

    # --- count of rows
    if op == "count" and (not val_cols or re.search(r"\b(rows?|records?|entries|items|orders|customers|people|employees)\b", q.lower()) or not _find_cols(t, q_stems, q, ("numeric",))):
        if group and not group.startswith("__"):
            c = defaultdict(int)
            for i in keep:
                c[str(t.rows[i][t.headers.index(group)])] += 1
            rows = sorted(c.items(), key=lambda kv: -kv[1])
            return _ok(f"There are {len(keep):,} rows{cond_txt}, split by {group}: " + ", ".join(f"{k} {v:,}" for k, v in rows[:6]) + ".", [[group, "Rows"]] + [[k, f"{v:,}"] for k, v in rows], "count", notes, len(keep), group, t, keep)
        return _ok(f"There are {len(keep):,} rows{cond_txt} in “{file_label}”" + (f" (sheet {t.name})" if len(tables) > 1 else "") + ".", [["Rows"], [f"{len(keep):,}"]], "count", notes, len(keep), None, t, keep)

    if not val_cols:
        return {"ok": False, "reason": "I couldn't tell which column the question refers to.", "columns": t.headers}
    # --- share of total by category
    if op == "share" and group or (op == "share" and cat_filters and val_cols):
        col = val_cols[0]
        base_keep = _row_mask(t, [], years, months, quarters, date_col)
        tot = sum(x for x in (t.num[col][i] for i in base_keep) if x is not None)
        part = sum(x for x in (t.num[col][i] for i in keep) if x is not None)
        if tot:
            _quality(t, col, keep, notes)
            return _ok(f"{part / tot:.1%} of total {col} ({_fmt(part)} of {_fmt(tot)}) comes from {' and '.join(f'{v}' for _, v in cat_filters) or 'the selected rows'}.",
                       [[col + " (selected)", col + " (all)", "Share"], [_fmt(part), _fmt(tot), f"{part / tot:.1%}"]], "share", notes, len(keep), col, t, keep)
    if op == "share" and not cat_filters:
        # share of rows per category of the grouping column
        if group and not group.startswith("__"):
            c = defaultdict(float)
            col = val_cols[0]
            for i in keep:
                v = t.num[col][i]
                if v is not None:
                    c[str(t.rows[i][t.headers.index(group)])] += v
            tot = sum(c.values())
            if tot:
                rows = sorted(c.items(), key=lambda kv: -kv[1])
                _quality(t, col, keep, notes)
                return _ok(f"Share of {col} by {group}: " + ", ".join(f"{k} {v / tot:.1%}" for k, v in rows[:6]) + ".", [[group, col, "Share"]] + [[k, _fmt(v), f"{v / tot:.1%}"] for k, v in rows], "share", notes, len(keep), col, t, keep)

    # --- change over time between the first and last period / two years
    if op == "change":
        col = val_cols[0]
        dc = date_col
        if not dc:
            tr = tabular.analyze_rows([t.headers] + t.rows, t.name)["trends"]
            tt = next((x for x in tr if x["column"] == col), None)
            if not tt:
                return {"ok": False, "reason": "There is no date or period column to measure change over.", "columns": t.headers}
        pairs = sorted(((t.dates[dc][i], t.num[col][i]) for i in keep if dc and t.dates[dc][i] and t.num[col][i] is not None), key=lambda x: x[0])
        if len(years) >= 2:                                   # compare two named years
            ysum = {y: sum(v for d, v in pairs if d.year == y) for y in years}
            a, b = years[0], years[-1]
            if ysum.get(a):
                ch = (ysum[b] - ysum[a]) / abs(ysum[a])
                _quality(t, col, keep, notes)
                return _ok(f"{col} {'rose' if ch > 0 else 'fell'} {abs(ch):.1%} from {a} ({_fmt(ysum[a])}) to {b} ({_fmt(ysum[b])}).",
                           [["Year", f"Total {col}"]] + [[str(y), _fmt(ysum[y])] for y in years], "change", notes, len(keep), col, t, keep)
        if len(pairs) >= 2:
            k = max(1, len(pairs) // 10)
            first, last = statistics.fmean(v for _, v in pairs[:k]), statistics.fmean(v for _, v in pairs[-k:])
            _quality(t, col, keep, notes)
            ch = None if first == 0 else (last - first) / abs(first)
            tr = tabular.analyze_rows([t.headers] + [t.rows[i] for i in keep], t.name)["trends"]
            tt = next((x for x in tr if x["column"] == col), None)
            word = "no clear trend" if tt and tt["direction"] == "flat" else (tt["direction"] if tt else "changing")
            s = (f"{col} is {word}: it went from about {_fmt(first)} at the start ({pairs[0][0].date()}) to about {_fmt(last)} at the end ({pairs[-1][0].date()})"
                 + (f", a change of {ch:+.1%}" if ch is not None else "") + f"{cond_txt}.")
            return _ok(s, [["Period", f"{col} (average)"], [f"first {k} row(s)", _fmt(first)], [f"last {k} row(s)", _fmt(last)]], "change", notes, len(pairs), col, t, keep)
        return {"ok": False, "reason": "Not enough dated rows to measure change.", "columns": t.headers}

    op = op or "sum"
    rank_groups = op in ("max", "min") and bool(group) and not re.search(r"\b(maximum|minimum|max|min|single)\b", q.lower())
    rank_dir = op
    if rank_groups:
        op = "sum"            # "which region had the highest revenue" = highest TOTAL revenue per region
    results = []
    for col in val_cols[:3]:
        _quality(t, col, keep, notes)
        if group and group.startswith("__period:"):
            unit = group.split(":")[1]
            dc = date_col
            if not dc:
                return {"ok": False, "reason": "There is no date column to group by.", "columns": t.headers}
            buckets = defaultdict(list)
            for i in keep:
                d, v = t.dates[dc][i], t.num[col][i]
                if d is not None and v is not None:
                    key = {"month": d.strftime("%Y-%m"), "year": str(d.year), "quarter": f"{d.year} Q{(d.month - 1) // 3 + 1}", "week": d.strftime("%Y-W%W"), "day": d.date().isoformat()}[unit]
                    buckets[key].append(v)
            rows = [(k, _agg(v, op)) for k, v in sorted(buckets.items())]
            if rank_groups:
                rows = sorted(rows, key=lambda kv: -(kv[1] or 0))
                if rank_dir == "min":
                    rows.reverse()
                rows = [(_period_label(k, unit), v) for k, v in rows]
            results.append((col, rows, group[9:]))
        elif group:
            buckets = defaultdict(list)
            for i in keep:
                v = t.num[col][i]
                if v is not None:
                    buckets[str(t.rows[i][t.headers.index(group)])].append(v)
            rows = sorted(((k, _agg(v, op)) for k, v in buckets.items()), key=lambda kv: -(kv[1] or 0), reverse=False)
            if rank_groups and rank_dir == "min":
                rows.reverse()
            results.append((col, rows, group))
        else:
            vals = [t.num[col][i] for i in keep if t.num[col][i] is not None]
            if not vals:
                continue
            results.append((col, [(None, _agg(vals, op))], None))
            if op in ("max", "min") and vals:
                target = max(vals) if op == "max" else min(vals)
                idx = next(i for i in keep if t.num[col][i] == target)
                results[-1] = (col, [(None, target)], "row", idx)
    if not results:
        return {"ok": False, "reason": f"The column “{val_cols[0]}” has no numeric values for those rows.", "columns": t.headers}
    verb = {"sum": "total", "avg": "average", "median": "median", "min": "lowest", "max": "highest", "count": "count", "std": "standard deviation"}[op]
    sentences, table = [], []
    for res in results:
        col, rows, grp = res[0], res[1], res[2]
        if grp == "row":
            idx = res[3]
            label = _row_label(t, idx, date_col)
            sentences.append(f"The {verb} {col} is {_fmt(rows[0][1])}{label}{cond_txt}.")
            table = [t.headers, [str(c) for c in t.rows[idx]]]
        elif grp and rank_groups:
            n = int(m.group(1)) if (m := re.search(r"\btop (\d+)\b", q.lower())) else 1
            word = "highest" if rank_dir == "max" else "lowest"
            lead = rows[:n]
            if n == 1:
                rest = f", ahead of {rows[1][0]} ({_fmt(rows[1][1])})" if len(rows) > 1 else ""
                sentences.append(f"{lead[0][0]} has the {word} total {col}{cond_txt}: {_fmt(lead[0][1])}{rest}.")
            else:
                sentences.append(f"The {n} {'highest' if rank_dir == 'max' else 'lowest'} by total {col}{cond_txt}: " + ", ".join(f"{k} ({_fmt(v)})" for k, v in lead) + ".")
            table = [[grp, f"Total {col}"]] + [[k, _fmt(v)] for k, v in rows]
        elif grp:
            top = rows[: (int(m.group(1)) if (m := re.search(r"\btop (\d+)\b", q.lower())) else 6)]
            sentences.append(f"{verb.capitalize()} {col} by {grp}{cond_txt}: " + ", ".join(f"{k} {_fmt(v)}" for k, v in top) + ("…" if len(rows) > len(top) else "") + ".")
            table = [[grp, f"{verb.capitalize()} {col}"]] + [[k, _fmt(v)] for k, v in rows]
        else:
            n = len([1 for i in keep if t.num[col][i] is not None])
            sentences.append(f"The {verb} {col} is {_fmt(rows[0][1])}{cond_txt}" + (f" (based on {n:,} rows)" if op in ("sum", "avg", "median") else "") + ".")
            table.extend([[f"{verb.capitalize()} {col}", _fmt(rows[0][1])]])
    if len(results) > 1 and not any(r[2] for r in results):
        table = [["Measure", "Value"]] + table
    res = _ok(" ".join(sentences), table, op, notes, len(keep), ", ".join(r[0] for r in results), t, keep)
    first = results[0]
    if len(results) == 1 and first[2] is None and first[1] and first[1][0][1] is not None:
        res["value"] = float(first[1][0][1])                       # a single scalar aggregate: can be re-computed independently
    elif len(results) == 1 and first[2] not in (None, "row") and op == "sum":
        res["groups"] = [(k, float(v)) for k, v in first[1] if v is not None]
    return res


def _period_label(key: str, unit: str) -> str:
    try:
        if unit == "month":
            return datetime.strptime(key, "%Y-%m").strftime("%B %Y")
    except ValueError:
        pass
    return key


def _row_label(t: Table, i: int, date_col: Optional[str]) -> str:
    if date_col and t.dates[date_col][i]:
        return f", on {t.dates[date_col][i].date()}"
    cats = [h for h in t.headers if t.types.get(h) == "category"]
    return f" ({cats[0]}: {t.rows[i][t.headers.index(cats[0])]})" if cats else ""


def _quality(t: Table, col: str, keep: List[int], notes: List[str]):
    missing = sum(1 for i in keep if t.num[col][i] is None)
    if missing:
        msg = f"{missing:,} of {len(keep):,} rows have an empty or non-numeric “{col}” value and were left out of the calculation."
        if msg not in notes:
            notes.append(msg)


def _ok(sentence, table, op, notes, rows_used, column, t: Table, keep: List[int]) -> dict:
    sample = [t.headers] + [[("" if c is None else str(c))[:40] for c in t.rows[i]] for i in keep[:5]]
    return {"ok": True, "sentence": sentence, "table": table, "operation": op, "notes": notes, "rows_used": rows_used, "column": column,
            "sheet": t.name, "source_rows": sample, "total_rows": len(t.rows), "columns": t.headers, "keep": list(keep), "column_first": (column or "").split(", ")[0] or None,
            "value": None, "groups": None}


def pick_tables(question: str, candidates: List[dict]) -> List[dict]:
    """Rank tabular files for a question by overlap between the question and their column names / sheet names /
    file name (candidates carry `columns`, `sheets`, `filename`)."""
    q_stems = {T._stem(t) for t in T.tokens(question)}
    out = []
    for c in candidates:
        cols = list(c.get("columns") or [])
        for sh in c.get("sheets") or []:
            cols += list(sh.get("columns") or []) + [sh.get("name", "")]
        sc = sum(_col_score(h, q_stems, question) for h in cols) + 0.5 * len(set(_norm(c["filename"])) & q_stems)
        out.append((sc, c))
    out.sort(key=lambda x: -x[0])
    return [c for sc, c in out if sc > 0]
