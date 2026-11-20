"""Real statistics over the ACTUAL contents of CSV/TSV/Excel files (re-read from disk, read-only, bounded).
Pure Python (csv + openpyxl): column typing, summary stats, date ranges, linear trends, correlations,
outliers. Every finding is computed from the data; nothing is estimated by a model."""
import csv
import io
import math
import re
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Sequence

MAX_ROWS = 200_000
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y",
                 "%b %Y", "%B %Y", "%Y-%m", "%d %b %Y", "%b %d, %Y", "%Y")
_NUM_CLEAN = re.compile(r"[,$€£\s]")


def _to_number(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = str(v).strip()
    if not s:
        return None
    pct = s.endswith("%")
    s = _NUM_CLEAN.sub("", s.rstrip("%"))
    if re.fullmatch(r"[-+]?\(?\d+(\.\d+)?\)?", s):
        neg = s.startswith("(") and s.endswith(")")
        try:
            x = float(s.strip("()"))
        except ValueError:
            return None
        return -x if neg else x
    return None


def _to_date(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    if hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day"):
        return datetime(v.year, v.month, v.day)
    if v is None:
        return None
    s = str(v).strip()
    if not s or len(s) > 30:
        return None
    for fmt in _DATE_FORMATS:
        try:
            d = datetime.strptime(s, fmt)
            if fmt == "%Y" and not (1900 <= d.year <= 2100):
                return None
            return d
        except ValueError:
            continue
    return None


def _read_csv(data: bytes) -> Dict[str, Dict]:
    text = data.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = []
    for i, row in enumerate(csv.reader(io.StringIO(text), dialect)):
        if i > MAX_ROWS:
            break
        if any(c.strip() for c in row):
            rows.append(row)
    return {"data": {"rows": rows}}


def _read_xlsx(data: bytes) -> Dict[str, Dict]:
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    out = {}
    try:
        for ws in wb.worksheets:
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i > MAX_ROWS:
                    break
                if any(c is not None and str(c).strip() for c in row):
                    rows.append(list(row))
            out[ws.title] = {"rows": rows}
    finally:
        wb.close()
    return out


def _unique_headers(raw: Sequence) -> List[str]:
    seen, out = {}, []
    for i, h in enumerate(raw):
        name = str(h).strip() if h is not None and str(h).strip() else f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


def _linreg(xs: List[float], ys: List[float]):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    r = sxy / math.sqrt(sxx * syy) if syy > 0 else 0.0
    return slope, my - slope * mx, r * r


def _corr(a: List[float], b: List[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def _fmt(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:,.1f}"
    return f"{x:,.3g}"


def analyze_rows(rows: List[list], name: str) -> Dict:
    if len(rows) < 2:
        return {"sheet": name, "rows": max(len(rows) - 1, 0), "columns": [], "findings": ["Too few rows to analyze."]}
    headers = _unique_headers(rows[0])
    body = [r + [None] * (len(headers) - len(r)) for r in rows[1:]]
    n_rows = len(body)
    cols, numeric, dates = [], {}, {}
    for ci, h in enumerate(headers):
        raw = [r[ci] if ci < len(r) else None for r in body]
        present = [v for v in raw if v is not None and str(v).strip() != ""]
        info = {"name": h, "missing": n_rows - len(present), "type": "text"}
        if not present:
            info["type"] = "empty"
            cols.append(info)
            continue
        nums = [(i, _to_number(v)) for i, v in enumerate(raw)]
        nvals = [(i, x) for i, x in nums if x is not None]
        dvals = [(i, _to_date(v)) for i, v in enumerate(raw)]
        dvals = [(i, d) for i, d in dvals if d is not None]
        looks_year = h.lower() in ("year", "yr")
        if len(nvals) >= 0.8 * len(present) and not (len(dvals) >= 0.8 * len(present) and not looks_year and len(nvals) < 0.8 * len(present)):
            vals = [x for _, x in nvals]
            info["type"] = "numeric"
            info.update(count=len(vals), min=min(vals), max=max(vals), mean=statistics.fmean(vals), median=statistics.median(vals),
                        std=statistics.pstdev(vals) if len(vals) > 1 else 0.0, sum=sum(vals))
            if len(vals) > 3 and info["std"] > 0:
                z = [abs(x - info["mean"]) / info["std"] for x in vals]
                info["outliers"] = sum(1 for v in z if v > 3.5)
            numeric[h] = nvals
        elif len(dvals) >= 0.8 * len(present):
            ds = [d for _, d in dvals]
            info["type"] = "date"
            info.update(min=min(ds).date().isoformat(), max=max(ds).date().isoformat())
            dates[h] = dvals
        else:
            from collections import Counter
            c = Counter(str(v).strip() for v in present)
            info["type"] = "categorical" if len(c) <= max(20, 0.2 * len(present)) else "text"
            info["unique"] = len(c)
            info["top_values"] = [{"value": k[:60], "count": v} for k, v in c.most_common(5)]
        cols.append(info)

    findings: List[str] = []
    trends = []
    # time axis: prefer a date column, else a 'year'-like numeric column, else row order
    axis_name, axis = None, None
    if dates:
        axis_name = max(dates, key=lambda k: len(dates[k]))
        axis = {i: d.timestamp() / 86400.0 for i, d in dates[axis_name]}
    else:
        for h in numeric:
            if h.lower() in ("year", "yr", "period", "month", "quarter", "week", "day"):
                axis_name, axis = h, {i: x for i, x in numeric[h]}
                break
    for h, pts in numeric.items():
        if h == axis_name or len(pts) < 4:
            continue
        if axis is not None:
            pairs = sorted(((axis[i], x) for i, x in pts if i in axis))
        else:
            pairs = [(float(i), x) for i, x in pts]
        if len(pairs) < 4:
            continue
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        slope, _, r2 = _linreg(xs, ys)
        k = max(1, len(ys) // 10)
        start, end = statistics.fmean(ys[:k]), statistics.fmean(ys[-k:])
        change = None if start == 0 else (end - start) / abs(start) * 100.0
        direction = "flat"
        if r2 >= 0.3 and (change is None or abs(change) >= 3):
            direction = "increasing" if slope > 0 else "decreasing"
        strength = "strong" if r2 >= 0.7 else "moderate" if r2 >= 0.4 else "weak"
        trends.append({"column": h, "direction": direction, "r2": round(r2, 3), "strength": strength,
                       "start_avg": start, "end_avg": end, "change_pct": None if change is None else round(change, 1),
                       "axis": axis_name or "row order", "points": len(pairs)})
    trends.sort(key=lambda t: -t["r2"])
    for t in trends[:6]:
        if t["direction"] == "flat":
            findings.append(f"{t['column']} shows no clear trend over {t['axis']} (R²={t['r2']}).")
        else:
            ch = f", {t['change_pct']:+.1f}% from the first to the last 10% of rows" if t["change_pct"] is not None else ""
            findings.append(f"{t['column']} is {t['direction']} over {t['axis']} ({t['strength']} fit, R²={t['r2']}{ch}; "
                            f"{_fmt(t['start_avg'])} → {_fmt(t['end_avg'])}).")
    # correlations
    corrs = []
    names = [h for h in numeric if h != axis_name]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ai, bj = dict(numeric[names[i]]), dict(numeric[names[j]])
            common = sorted(set(ai) & set(bj))
            if len(common) >= 8:
                r = _corr([ai[k] for k in common], [bj[k] for k in common])
                if abs(r) >= 0.6:
                    corrs.append({"a": names[i], "b": names[j], "r": round(r, 3), "n": len(common)})
    corrs.sort(key=lambda c: -abs(c["r"]))
    for c in corrs[:4]:
        findings.append(f"{c['a']} and {c['b']} are {'positively' if c['r'] > 0 else 'negatively'} correlated (r={c['r']}, n={c['n']}).")
    for c in cols:
        if c["type"] == "numeric" and c.get("outliers"):
            findings.append(f"{c['name']} has {c['outliers']} extreme value(s) (>3.5σ from the mean).")
        if c.get("missing", 0) > 0.2 * n_rows and c["type"] != "empty":
            findings.append(f"{c['name']} is missing in {c['missing']} of {n_rows} rows ({c['missing'] / n_rows:.0%}).")
    for c in cols:
        if c["type"] == "categorical" and c.get("top_values"):
            tv = c["top_values"][0]
            findings.append(f"Most common {c['name']}: '{tv['value']}' ({tv['count']} of {n_rows} rows).")
            break
    if not findings:
        findings.append("No notable trends, correlations or outliers were detected.")
    return {"sheet": name, "rows": n_rows, "columns": cols, "trends": trends[:10], "correlations": corrs[:6],
            "time_axis": axis_name, "findings": findings}


def analyze_file(data: bytes, ext: str) -> Dict:
    ext = ext.lower()
    if ext in ("xlsx", "xlsm"):
        sheets = _read_xlsx(data)
    elif ext in ("csv", "tsv"):
        sheets = _read_csv(data)
    else:
        raise ValueError("not a tabular file")
    results = [analyze_rows(s["rows"], name) for name, s in sheets.items()]
    return {"sheets": results, "findings": [f"[{r['sheet']}] {f}" if len(results) > 1 else f for r in results for f in r["findings"]][:14]}


def preview_rows(data: bytes, ext: str, limit: int = 30):
    """First rows of the first sheet for the file preview (real data)."""
    ext = ext.lower()
    sheets = _read_xlsx(data) if ext in ("xlsx", "xlsm") else _read_csv(data)
    out = []
    for name, s in list(sheets.items())[:3]:
        rows = s["rows"][:limit + 1]
        out.append({"sheet": name, "total_rows": len(s["rows"]) - 1,
                    "rows": [["" if c is None else str(c)[:80] for c in r[:30]] for r in rows]})
    return out
