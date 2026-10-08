"""
Controlled, offline LLM providers for the benchmark (no network, ever).

`ScriptedProvider` implements the production `LLMProvider` protocol
(`name` + `complete(system_prompt, user_prompt) -> str`) and answers each of
the four LLM stages with a *predefined, deterministic policy* that only reads
what the pipeline itself puts in the prompt. It never sees ground truth.

IT IS NOT A MODEL. Results under variant D therefore measure how the
architecture behaves under *controlled* LLM assistance (do the gates fire, are
valid outputs applied safely, do invalid ones fall back) -- not how good a real
LLM is. The policies are deliberately simple and documented:

  goal       -> adds the SYNONYMS-mapped tokens for words in the query
  expansion  -> same SYNONYMS map, only for words not already present
  comparison -> picks the listed candidate whose filename/tags/summary share
                the most words with the query (ties -> first listed, i.e. the
                deterministic order stands); confidence 0.8, or 0.4 when there
                is zero overlap (below the production 0.5 floor, so it is
                ignored -- exercising the low_confidence path)
  explanation-> a one-sentence template quoting the filename it was shown

Modes: "useful" | "unavailable" (raises LLMUnavailableError) | "invalid"
(returns non-JSON text). `fail_stages` makes only the named stages
unavailable (used to prove explanation never influences ranking).

`CountingProvider` wraps any provider from OUTSIDE the production path and
records, per stage: attempts (complete() was invoked) vs calls (it returned a
response -- i.e. a call that would have been billed). A NullProvider /
unavailable provider yields attempts > 0 but calls == 0.
"""
import ast
import json
import re
from typing import Dict, Iterable, Optional

from app.llm import candidate_comparer, goal_interpreter, query_expander, result_explainer
from app.llm.provider import LLMUnavailableError

STAGE_PROMPTS = {
    "goal": goal_interpreter.SYSTEM_PROMPT,
    "expansion": query_expander.SYSTEM_PROMPT,
    "comparison": candidate_comparer.SYSTEM_PROMPT,
    "explanation": result_explainer.SYSTEM_PROMPT,
}
STAGES = tuple(STAGE_PROMPTS)

# Documented, small, generic vocabulary map (word in query -> related terms).
SYNONYMS = {
    "financial": ["finance"],
    "finances": ["finance"],
    "money": ["finance", "budget"],
    "spending": ["budget"],
    "cost": ["budget"],
    "costs": ["budget"],
}
_WORD = re.compile(r"[a-z0-9]+")


def identify_stage(system_prompt: str) -> str:
    for stage, prompt in STAGE_PROMPTS.items():
        if system_prompt == prompt:
            return stage
    return "other"


def _query_words(user_prompt: str):
    m = re.search(r"User query: (.*)\n", user_prompt)
    if not m:
        return []
    try:
        q = ast.literal_eval(m.group(1).strip())
    except (ValueError, SyntaxError):
        q = m.group(1)
    return _WORD.findall(str(q).lower())


def _json_after_first_line(user_prompt: str) -> dict:
    return json.loads(user_prompt.split("\n", 1)[1])


def _synonym_terms(words: Iterable[str], exclude: Iterable[str] = ()) -> list:
    exclude = set(exclude)
    out = []
    for w in words:
        for t in SYNONYMS.get(w, []):
            if t not in exclude and t not in out:
                out.append(t)
    return out


class ScriptedProvider:
    def __init__(self, mode: str = "useful", fail_stages: Iterable[str] = ()):
        if mode not in ("useful", "unavailable", "invalid"):
            raise ValueError(mode)
        self.mode = mode
        self.fail_stages = set(fail_stages)
        self.name = f"scripted:{mode}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        stage = identify_stage(system_prompt)
        if self.mode == "unavailable" or stage in self.fail_stages:
            raise LLMUnavailableError(f"scripted provider unavailable (stage={stage})")
        if self.mode == "invalid":
            return "Sure! I think the best answer is probably the budget file."
        handler = getattr(self, f"_{stage}", None)
        if handler is None:
            raise LLMUnavailableError(f"scripted provider has no policy for stage {stage!r}")
        return handler(user_prompt)

    # --- policies ----------------------------------------------------------
    def _goal(self, user_prompt: str) -> str:
        tokens = _synonym_terms(_query_words(user_prompt))
        return json.dumps({
            "file_type": None, "topic_tokens": tokens, "filename_candidate": None,
            "retrieval_preference": "lexical", "confidence": 0.8,
            "rationale": "Scripted benchmark interpretation (synonym map only).",
        })

    def _expansion(self, user_prompt: str) -> str:
        words = _query_words(user_prompt)
        terms = _synonym_terms(words, exclude=words)
        return json.dumps({
            "expanded_terms": terms, "confidence": 0.7 if terms else 0.3,
            "rationale": "Scripted benchmark expansion (synonym map only).",
        })

    def _comparison(self, user_prompt: str) -> str:
        payload = _json_after_first_line(user_prompt)
        q_words = set(_WORD.findall(str(payload.get("user_query", "")).lower()))
        best_id, best_overlap = None, -1
        for cand in payload["candidates"]:
            text = " ".join([
                str(cand.get("filename") or "").replace("_", " "),
                " ".join(cand.get("tags") or []),
                str(cand.get("summary") or ""),
            ]).lower()
            overlap = len(q_words & set(_WORD.findall(text)))
            if overlap > best_overlap:  # strict > : ties keep the earlier (deterministic) order
                best_id, best_overlap = cand["file_id"], overlap
        return json.dumps({
            "selected_file_id": best_id,
            "confidence": 0.8 if best_overlap > 0 else 0.4,
            "rationale": f"Scripted lexical-overlap judgement (overlap={best_overlap}).",
        })

    def _explanation(self, user_prompt: str) -> str:
        evidence = _json_after_first_line(user_prompt)
        result = evidence.get("result") or {}
        return json.dumps({
            "summary": f"'{result.get('filename')}' is the closest match to the request based on the retrieved evidence.",
            "evidence_points": [f"File type: {result.get('file_type')}"],
            "confidence": 0.8,
        })


class CountingProvider:
    """Transparent wrapper: counts attempts/calls per stage, changes nothing."""

    def __init__(self, inner):
        self._inner = inner
        self.name = getattr(inner, "name", "unknown")
        self.attempts: Dict[str, int] = {s: 0 for s in (*STAGES, "other")}
        self.calls: Dict[str, int] = {s: 0 for s in (*STAGES, "other")}

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        stage = identify_stage(system_prompt)
        self.attempts[stage] += 1
        response = self._inner.complete(system_prompt, user_prompt)  # exceptions propagate unchanged
        self.calls[stage] += 1
        return response

    @property
    def total_attempts(self) -> int:
        return sum(self.attempts.values())

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())
