"""
Structured LLM goal-interpretation schema -- Phase 5A
(architecture_final.md §9: "Intent extraction fallback, when the
deterministic heuristic is inconclusive"; prompt requirement 3: "The LLM
must return a validated structured object, NOT SQL, arbitrary Python, or
unrestricted tool calls.").

The schema is deliberately narrow: every field is a whitelisted, typed,
bounded value (an enum-like file_type, a short list of plain lowercase
tokens, a bounded confidence float, a short rationale string). There is no
field capable of holding SQL, Python, or a tool-call payload -- the model
literally has nowhere to put one. That's the actual enforcement mechanism,
not a runtime "reject if it looks like SQL" content filter (which would be
guessable/bypassable). `model_config = ConfigDict(extra="forbid")`
additionally rejects any key the LLM tries to add beyond this whitelist
(e.g. an invented "sql" or "tool_call" field fails validation outright).

Nothing in this schema can express `intent` (the system has exactly one:
"file_discovery" -- there is no second intent in this architecture for an
LLM to choose between) or `user_scope`/access control. Both stay entirely
under deterministic control (app/scope.py, app/goal.py) and are never
influenced by LLM output, in this phase or any later one.

Phase 5B adds `QueryExpansion` (architecture_final.md §9: "query_expansion,
to reformulate a query that returned weak lexical/semantic evidence") --
same narrow-whitelist philosophy: a short bounded list of plain search
phrases, a confidence, a short rationale. No file IDs, user IDs, tool
calls, or ranking scores are representable -- there's no field for any of
them, and (belt-and-suspenders) a term that's purely numeric is rejected
outright since it would look like a database ID rather than a search
phrase.
"""
import json
import re
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

ALLOWED_FILE_TYPES = {"pptx", "xlsx", "docx", "pdf", "png"}
ALLOWED_RETRIEVAL_PREFERENCES = {"metadata", "lexical", "semantic", "hybrid"}
MAX_TOPIC_TOKENS = 8
MAX_TOKEN_LENGTH = 40
MAX_RATIONALE_LENGTH = 200
MAX_EXPANDED_TERMS = 6
MAX_EXPANDED_TERM_LENGTH = 60
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_ ]*")


class LLMResponseInvalidError(ValueError):
    """Raised when the LLM's raw output is not valid JSON, is not a single
    JSON object, or fails schema validation. Callers (app/llm/
    goal_interpreter.py, app/llm/query_expander.py) must catch this and
    fall back to deterministic behavior unchanged -- never partially trust
    a payload that failed validation, and never retry the LLM in a loop."""


def _parse_json_object(raw_text: str) -> dict:
    """Shared by both parse_* functions below. Tolerates a fenced
    ```json ... ``` block (a common model habit) but otherwise requires
    the ENTIRE response to be exactly one JSON object -- no free text, no
    "find the first {...} in the string" scanning. That leniency is
    exactly what would let a model smuggle extra prose or instructions
    past validation, so it's deliberately not supported."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMResponseInvalidError(f"LLM output is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise LLMResponseInvalidError("LLM output must be a single JSON object")

    return payload


class LLMGoalInterpretation(BaseModel):
    """The ONLY shape an LLM goal interpretation may take. Every field is
    advisory input to a separate, deterministic merge step
    (goal_interpreter._merge_llm_interpretation) -- nothing on this model
    is ever executed, used as a query fragment, or capable of changing
    `user_scope`, retrieval algorithms, or ranking."""

    model_config = ConfigDict(extra="forbid")

    file_type: Optional[str] = None
    topic_tokens: List[str] = Field(default_factory=list)
    filename_candidate: Optional[str] = None
    retrieval_preference: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = Field(default=None, max_length=MAX_RATIONALE_LENGTH)

    @field_validator("file_type")
    @classmethod
    def _validate_file_type(cls, v):
        if v is None:
            return v
        v = str(v).strip().lower()
        if v not in ALLOWED_FILE_TYPES:
            raise ValueError(f"file_type must be one of {sorted(ALLOWED_FILE_TYPES)}, got {v!r}")
        return v

    @field_validator("retrieval_preference")
    @classmethod
    def _validate_retrieval_preference(cls, v):
        if v is None:
            return v
        v = str(v).strip().lower()
        if v not in ALLOWED_RETRIEVAL_PREFERENCES:
            raise ValueError(
                f"retrieval_preference must be one of {sorted(ALLOWED_RETRIEVAL_PREFERENCES)}, got {v!r}"
            )
        return v

    @field_validator("topic_tokens")
    @classmethod
    def _validate_topic_tokens(cls, v):
        if len(v) > MAX_TOPIC_TOKENS:
            raise ValueError(f"topic_tokens: at most {MAX_TOPIC_TOKENS} allowed, got {len(v)}")
        cleaned = []
        for tok in v:
            if not isinstance(tok, str):
                raise ValueError("topic_tokens must all be strings")
            tok = tok.strip().lower()
            if not tok:
                continue
            if len(tok) > MAX_TOKEN_LENGTH:
                raise ValueError(f"topic token too long (max {MAX_TOKEN_LENGTH} chars): {tok!r}")
            if not _TOKEN_RE.fullmatch(tok):
                raise ValueError(f"topic token contains disallowed characters: {tok!r}")
            cleaned.append(tok)
        return cleaned

    @field_validator("filename_candidate")
    @classmethod
    def _validate_filename_candidate(cls, v):
        if v is None:
            return v
        v = str(v).strip()
        if not v or len(v) > 200:
            raise ValueError("filename_candidate must be a non-empty string up to 200 chars")
        return v


def parse_llm_goal_response(raw_text: str) -> LLMGoalInterpretation:
    """Parses + validates a raw LLM response into an `LLMGoalInterpretation`."""
    payload = _parse_json_object(raw_text)
    try:
        return LLMGoalInterpretation.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseInvalidError(f"LLM output failed schema validation: {exc}") from exc


class QueryExpansion(BaseModel):
    """The ONLY shape an LLM query expansion may take -- Phase 5B. Every
    field is advisory input to deterministic query-building code
    (app/planning.py) -- nothing here is ever executed, matched directly
    against a file ID, or capable of changing `user_scope`, retrieval
    algorithms, or ranking. There is no `tool`/`action` field: the LLM
    proposes search text only, it never selects or invokes a tool."""

    model_config = ConfigDict(extra="forbid")

    expanded_terms: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = Field(default=None, max_length=MAX_RATIONALE_LENGTH)

    @field_validator("expanded_terms")
    @classmethod
    def _validate_expanded_terms(cls, v):
        if len(v) > MAX_EXPANDED_TERMS:
            raise ValueError(f"expanded_terms: at most {MAX_EXPANDED_TERMS} allowed, got {len(v)}")
        cleaned = []
        for term in v:
            if not isinstance(term, str):
                raise ValueError("expanded_terms must all be strings")
            term = term.strip().lower()
            if not term:
                continue
            if len(term) > MAX_EXPANDED_TERM_LENGTH:
                raise ValueError(f"expanded term too long (max {MAX_EXPANDED_TERM_LENGTH} chars): {term!r}")
            if not _TOKEN_RE.fullmatch(term):
                raise ValueError(f"expanded term contains disallowed characters: {term!r}")
            bare = term.replace(" ", "").replace("-", "").replace("_", "")
            if bare.isdigit():
                raise ValueError(
                    f"expanded term looks like a numeric ID, not a search phrase: {term!r}"
                )
            cleaned.append(term)
        # de-duplicate, preserve order
        seen = set()
        deduped = []
        for t in cleaned:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped


def parse_query_expansion_response(raw_text: str) -> QueryExpansion:
    """Parses + validates a raw LLM response into a `QueryExpansion`."""
    payload = _parse_json_object(raw_text)
    try:
        return QueryExpansion.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseInvalidError(f"LLM output failed schema validation: {exc}") from exc


class CandidateComparison(BaseModel):
    """The ONLY shape an LLM candidate comparison may take -- Phase 5C.
    The model names exactly one preferred candidate (`selected_file_id`),
    a confidence, and a short rationale (a concise decision string, never
    chain-of-thought). Whether that ID is one of the candidates actually
    supplied is enforced by app/llm/candidate_comparer.py, which owns the
    supplied set; the schema itself only guarantees the type: `StrictInt`
    rejects strings, floats and booleans, so `"12"`, `12.0` and `true`
    all fail validation. There is no field for a tool call, SQL, a new
    candidate, a score, or scope -- nothing to smuggle one into.
    """

    model_config = ConfigDict(extra="forbid")

    selected_file_id: StrictInt
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = Field(default=None, max_length=MAX_RATIONALE_LENGTH)


def parse_candidate_comparison_response(raw_text: str) -> CandidateComparison:
    """Parses + validates a raw LLM response into a `CandidateComparison`."""
    payload = _parse_json_object(raw_text)
    try:
        return CandidateComparison.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseInvalidError(f"LLM output failed schema validation: {exc}") from exc


MAX_EXPLANATION_SUMMARY_LENGTH = 400
MAX_EVIDENCE_POINTS = 5
MAX_EVIDENCE_POINT_LENGTH = 200


class ResultExplanation(BaseModel):
    """The ONLY shape an LLM result explanation may take -- Phase 5D.
    Pure presentation text: a short summary, a few concrete evidence
    statements and a confidence. There is deliberately NO `file_id`,
    ranking, score, tool or scope field -- the explanation is bound to the
    already-selected result by the graph (state), never by the model, so
    the model has no way to name, change or introduce a result.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=MAX_EXPLANATION_SUMMARY_LENGTH)
    evidence_points: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("summary")
    @classmethod
    def _validate_summary(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("summary must not be blank")
        return v

    @field_validator("evidence_points")
    @classmethod
    def _validate_evidence_points(cls, v):
        if len(v) > MAX_EVIDENCE_POINTS:
            raise ValueError(f"evidence_points: at most {MAX_EVIDENCE_POINTS} allowed, got {len(v)}")
        cleaned = []
        for point in v:
            if not isinstance(point, str):
                raise ValueError("evidence_points must all be strings")
            point = point.strip()
            if not point:
                continue
            if len(point) > MAX_EVIDENCE_POINT_LENGTH:
                raise ValueError(f"evidence point too long (max {MAX_EVIDENCE_POINT_LENGTH} chars)")
            cleaned.append(point)
        return cleaned


def parse_result_explanation_response(raw_text: str) -> ResultExplanation:
    """Parses + validates a raw LLM response into a `ResultExplanation`."""
    payload = _parse_json_object(raw_text)
    try:
        return ResultExplanation.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseInvalidError(f"LLM output failed schema validation: {exc}") from exc
