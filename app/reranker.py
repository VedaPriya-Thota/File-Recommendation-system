"""
Cross-encoder reranking provider abstraction (Phase 3) --
architecture_final.md §6 `rerank_candidates`.

PRODUCTION / INTENDED PROVIDER
    `sentence-transformers` CrossEncoder, model
    `cross-encoder/ms-marco-MiniLM-L-6-v2`. This is what `requirements.txt`
    already declares (sentence-transformers, added in Phase 2B for
    embeddings) and what `get_reranker_provider()` tries FIRST, every time.
    In any environment with normal internet access (or a pre-populated
    `~/.cache/huggingface`), this is the provider that loads and the rest
    of this module is dead code.

SANDBOX DEVIATION (read this before assuming a bug -- same story as
`app/embeddings.py`'s module docstring, applied here)
    This development sandbox's network egress policy blocks
    huggingface.co outright, so `sentence-transformers` cannot download
    `cross-encoder/ms-marco-MiniLM-L-6-v2` here. `get_reranker_provider()`
    falls back to a small deterministic, dependency-free lexical-overlap
    scorer (`_LexicalOverlapProvider`) so `rerank_candidates`'s mechanics
    (candidate-set closure, evidence selection, ordering, candidate-pool
    integration) remain exercisable end-to-end even without a
    network-reachable cross-encoder. This fallback is NOT a cross-encoder
    and its scores are NOT calibrated probabilities -- it exists purely so
    the surrounding plumbing is testable here. Swapping in a real
    network-reachable environment requires no code change:
    `get_reranker_provider()` unconditionally tries the real CrossEncoder
    first.

Similarity/scoring is always query+candidate-text pairwise scoring (a
cross-encoder scores a (query, text) pair jointly, unlike the embeddings
module's bi-encoder cosine similarity) -- this is architecturally a
different kind of model from `app/embeddings.py`'s, which is why it gets
its own small abstraction rather than reusing that module's provider.
"""
import logging
import re
from typing import List, Optional, Protocol

logger = logging.getLogger("file_discovery_agent.reranker")

PREFERRED_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_WORD_RE = re.compile(r"[a-z0-9]+")


class RerankProvider(Protocol):
    name: str

    def score(self, query: str, texts: List[str]) -> List[float]: ...


class _CrossEncoderProvider:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # local import: optional dep

        self._model = CrossEncoder(model_name)
        self.name = f"cross-encoder:{model_name}"

    def score(self, query: str, texts: List[str]) -> List[float]:
        pairs = [(query, t or "") for t in texts]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


class _LexicalOverlapProvider:
    """Deterministic, dependency-free fallback -- see module docstring's
    SANDBOX DEVIATION section. NOT a cross-encoder: a normalized
    query/text token-overlap score, used only when no real cross-encoder
    model can be loaded. Scores are in [0, 1] but are plain token-overlap
    ratios, not a learned relevance judgment."""

    def __init__(self):
        self.name = "lexical_overlap_fallback (NOT a cross-encoder -- see reranker.py docstring)"

    def score(self, query: str, texts: List[str]) -> List[float]:
        q_tokens = set(_WORD_RE.findall((query or "").lower()))
        out = []
        for t in texts:
            t_tokens = set(_WORD_RE.findall((t or "").lower()))
            if not q_tokens or not t_tokens:
                out.append(0.0)
                continue
            overlap = len(q_tokens & t_tokens)
            out.append(overlap / len(q_tokens))
        return out


_provider: Optional[RerankProvider] = None


def get_reranker_provider() -> RerankProvider:
    global _provider
    if _provider is not None:
        return _provider

    try:
        _provider = _CrossEncoderProvider(PREFERRED_MODEL_NAME)
        logger.info("reranker provider: %s", _provider.name)
        return _provider
    except Exception as exc:  # noqa: BLE001 -- any failure means "fall back"
        logger.warning("cross-encoder unavailable (%s); using lexical-overlap fallback", exc)

    _provider = _LexicalOverlapProvider()
    logger.info("reranker provider: %s", _provider.name)
    return _provider


def score_pairs(query: str, texts: List[str]) -> List[float]:
    return get_reranker_provider().score(query, texts)


def get_reranker_provider_name() -> str:
    return get_reranker_provider().name


def reset_provider_for_tests() -> None:
    """Test-only hook: forces the next `score_pairs`/`get_reranker_provider`
    call to re-select a provider. Mirrors the module-level caching pattern
    in `app/embeddings.py`, which needs the same reset for isolated
    provider-selection tests."""
    global _provider
    _provider = None
