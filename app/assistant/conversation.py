"""Short-term conversation memory (in process). Keeps just what is needed to resolve follow-ups such as
"What are its main limitations?": which files the previous answer was about and what the topic was."""
import threading
import time
import uuid
from typing import Dict, List, Optional

_LOCK = threading.Lock()
_CONV: Dict[str, dict] = {}
MAX_TURNS = 6
TTL_S = 6 * 3600


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def get(conv_id: Optional[str]) -> dict:
    now = time.time()
    with _LOCK:
        for k in [k for k, v in _CONV.items() if now - v["touched"] > TTL_S]:
            _CONV.pop(k, None)
        if conv_id and conv_id in _CONV:
            _CONV[conv_id]["touched"] = now
            return _CONV[conv_id]
        cid = conv_id or new_id()
        _CONV[cid] = {"id": cid, "turns": [], "focus_file_ids": [], "focus_names": [], "topic_terms": [], "touched": now}
        return _CONV[cid]


def record(conv: dict, question: str, kind: str, file_ids: List[int], names: List[str], terms: List[str], answered: bool) -> None:
    with _LOCK:
        conv["turns"].append({"q": question, "kind": kind, "files": file_ids, "answered": answered})
        conv["turns"] = conv["turns"][-MAX_TURNS:]
        if file_ids:                      # only move the focus when the answer was actually about some files
            conv["focus_file_ids"], conv["focus_names"] = list(file_ids), list(names)
        if terms:
            conv["topic_terms"] = list(terms)[:8]
        conv["touched"] = time.time()
