"""
Ablation variants. Each is expressed ONLY through existing production hooks --
`run_query(max_iterations=..., registry=..., llm_provider=...)` -- so no
production code is modified and no retrieval implementation is faked.

A  single_shot_metadata  max_iterations=1. The planner's first action is always
                         metadata_search, and reflect() force-finishes at the
                         iteration cap, so this is a one-shot metadata_search
                         followed by the normal deterministic final ranking. No
                         escalation, no replanning. (A "metadata-only registry"
                         was rejected: ToolRegistry.get() raises an uncaught
                         KeyError for an unregistered tool, so the first
                         escalation would crash instead of measuring anything.)
B  agentic_deterministic Full loop + metadata/BM25/semantic/hybrid/rerank, no
                         LLM (NullProvider), personalization DISABLED.
C  agentic_personalized  B + the real get_user_activity tool reading the
                         explicit telemetry fixture.
D  full_system           C + controlled ScriptedProvider("useful") for 5A-5D.

Supplementary variants for behavioural scenarios:
D_llm_unavailable        D's configuration with a provider that always raises.
D_llm_invalid_output     D's configuration with a provider returning non-JSON.
D_no_explanation         D with only the explanation stage unavailable (proves
                         5D never influences ranking).
B/C/D_cap3_diagnostic    B/C/D with max_iterations=3 -- forces flat pools to finish
                         so finish-time stages become observable (diagnostic only).

Personalization is disabled in A/B by swapping `get_user_activity` for a
pass-through `null_personalization_tool` that returns the supplied candidates
with affinity 0.0. Control flow (when the planner calls it, iteration counts)
is therefore IDENTICAL between B and C; only the ranking evidence differs --
and corpus, user, query and candidate scope are the same objects in both.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from app.scope import UserScope
from app.tools import build_registry
from app.tools.registry import Tool, ToolRegistry, ToolResult

from evaluation.providers import CountingProvider, ScriptedProvider

PERSONALIZATION_TOOL = "get_user_activity"


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    max_iterations: Optional[int]  # None -> production default
    personalization: bool
    llm_mode: Optional[str]  # None -> NullProvider (no LLM); else ScriptedProvider mode
    fail_stages: Tuple[str, ...] = ()
    core: bool = True  # part of the A-D ablation (vs. supplementary scenario variants)


VARIANTS = (
    Variant("A_single_shot_metadata", "One metadata_search, no replanning, no LLM, no personalization", 1, False, None),
    Variant("B_agentic_deterministic", "Full deterministic agent loop; personalization disabled; no LLM", None, False, None),
    Variant("C_agentic_personalized", "B plus deterministic personalization from telemetry; no LLM", None, True, None),
    Variant("D_full_system", "C plus controlled scripted LLM goal/expansion/comparison/explanation", None, True, "useful"),
    Variant("D_llm_unavailable", "D configuration, provider always unavailable", None, True, "unavailable", core=False),
    Variant("D_llm_invalid_output", "D configuration, provider returns invalid output", None, True, "invalid", core=False),
    Variant("D_no_explanation", "D configuration, explanation stage unavailable only", None, True, "useful", ("explanation",), core=False),
    # DIAGNOSTIC ONLY. max_iterations=3 equals the length of the metadata -> rerank -> get_user_activity
    # chain, so flat pools are force-finished (reflect's existing iteration-cap rule) instead of clarified.
    # This exposes the finish-time stages (personalized final ranking, Phase 5C) that the default flow never
    # reaches on this corpus. It uses an existing run_query parameter; it is NOT a proposal for a new default.
    Variant("B_cap3_diagnostic", "B with max_iterations=3 (diagnostic)", 3, False, None, core=False),
    Variant("C_cap3_diagnostic", "C with max_iterations=3 (diagnostic)", 3, True, None, core=False),
    Variant("D_cap3_diagnostic", "D with max_iterations=3 (diagnostic)", 3, True, "useful", core=False),
)
VARIANTS_BY_NAME = {v.name: v for v in VARIANTS}
CORE_VARIANT_NAMES = tuple(v.name for v in VARIANTS if v.core)


def null_personalization(args: dict, user_scope: UserScope) -> ToolResult:
    out = []
    for c in args.get("candidates") or []:
        out.append({"file_id": c["file_id"], "metadata": dict(c.get("metadata") or {}),
                    "score": 0.0, "source": PERSONALIZATION_TOOL})
    return ToolResult(tool=PERSONALIZATION_TOOL, candidates=out)


def build_variant_registry(variant: Variant) -> ToolRegistry:
    full = build_registry()
    if variant.personalization:
        return full
    reg = ToolRegistry()
    for name in full.names():
        if name == PERSONALIZATION_TOOL:
            reg.register(Tool(name=name, description="benchmark: personalization disabled (affinity 0.0)",
                              fn=null_personalization))
        else:
            reg.register(full.get(name))
    return reg


def build_variant_provider(variant: Variant) -> CountingProvider:
    from app.llm.provider import NullProvider

    inner = NullProvider() if variant.llm_mode is None else ScriptedProvider(variant.llm_mode, variant.fail_stages)
    return CountingProvider(inner)
