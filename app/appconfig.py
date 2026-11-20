"""
Runtime application settings (stored in the `app_settings` table, one JSON document under key "settings").

Everything here is user-editable from the Settings page. Defaults are privacy-preserving: nothing leaves the
machine unless the user explicitly picks an external model provider AND consents to sending excerpts to it.
"""
import copy
import json
import threading
from datetime import datetime, timezone

from app.db.connection import get_cursor
from app.ingestion.extract import SUPPORTED_EXTENSIONS
from app.ingestion.source import DEFAULT_EXCLUSIONS

_LOCK = threading.Lock()
_CACHE = {"v": None}

DEFAULTS = {
    "indexing": {
        "exclusions": list(DEFAULT_EXCLUSIONS),
        "include_hidden": False,
        "max_file_mb": 25,
        "enabled_formats": list(SUPPORTED_EXTENSIONS),
        "semantic_enabled": True,
        "watch_enabled": True,
        "watch_interval_s": 60,
    },
    "privacy": {
        "allow_external_llm": False,   # consent gate: excerpts are sent to a non-local model only if True
        "log_queries": True,           # keep assistant history (local DB) for the Recommendations page
    },
    "model": {
        "provider": "none",            # none | anthropic | openai_compatible
        "model": "",
        "base_url": "",                # openai_compatible: e.g. http://localhost:11434/v1 (Ollama)
        "timeout_s": 60,
        "max_tokens": 1200,
    },
    "assistant": {
        "max_files_analyzed": 5,
        "excerpts_per_file": 3,
        "excerpt_chars": 500,
    },
}

LIMITS = {
    ("indexing", "max_file_mb"): (1, 500),
    ("indexing", "watch_interval_s"): (10, 3600),
    ("model", "timeout_s"): (5, 600),
    ("model", "max_tokens"): (128, 8000),
    ("assistant", "max_files_analyzed"): (1, 10),
    ("assistant", "excerpts_per_file"): (1, 6),
    ("assistant", "excerpt_chars"): (150, 1500),
}


def _merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for section, values in (patch or {}).items():
        if section in out and isinstance(values, dict):
            for k, v in values.items():
                if k in out[section]:
                    out[section][k] = v
    return out


def _validate(s: dict) -> dict:
    for (sec, key), (lo, hi) in LIMITS.items():
        try:
            v = int(s[sec][key])
        except (TypeError, ValueError):
            v = DEFAULTS[sec][key]
        s[sec][key] = max(lo, min(hi, v))
    idx = s["indexing"]
    idx["exclusions"] = [str(x).strip() for x in idx["exclusions"] if str(x).strip()][:200]
    idx["enabled_formats"] = sorted({str(x).lower().lstrip(".") for x in idx["enabled_formats"]} & set(SUPPORTED_EXTENSIONS))
    for k in ("include_hidden", "semantic_enabled", "watch_enabled"):
        idx[k] = bool(idx[k])
    s["privacy"] = {k: bool(v) for k, v in s["privacy"].items()}
    if s["model"]["provider"] not in ("none", "anthropic", "openai_compatible"):
        s["model"]["provider"] = "none"
    s["model"]["model"] = str(s["model"]["model"] or "").strip()[:200]
    s["model"]["base_url"] = str(s["model"]["base_url"] or "").strip()[:300]
    return s


def get_settings(refresh: bool = False) -> dict:
    with _LOCK:
        if _CACHE["v"] is not None and not refresh:
            return copy.deepcopy(_CACHE["v"])
        stored = {}
        try:
            with get_cursor() as cur:
                cur.execute("SELECT value FROM app_settings WHERE key = %s;", ("settings",))
                row = cur.fetchone()
                if row:
                    stored = json.loads(row["value"])
        except Exception:  # noqa: BLE001 -- table missing (legacy DB) or unreadable: defaults
            stored = {}
        merged = _validate(_merge(DEFAULTS, stored))
        _CACHE["v"] = merged
        return copy.deepcopy(merged)


def update_settings(patch: dict) -> dict:
    current = get_settings()
    new = _validate(_merge(current, patch))
    now = datetime.now(timezone.utc).isoformat()
    with get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM app_settings WHERE key = %s;", ("settings",))
        cur.execute("INSERT INTO app_settings (key, value, updated_at) VALUES (%s, %s, %s);", ("settings", json.dumps(new), now))
    with _LOCK:
        _CACHE["v"] = new
    return copy.deepcopy(new)


def reset_settings() -> dict:
    with get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM app_settings WHERE key = %s;", ("settings",))
    with _LOCK:
        _CACHE["v"] = None
    return get_settings(refresh=True)
