"""
Embedding provider abstraction (Phase 2B) -- architecture_final.md §6/§10
future semantic layer.

PRODUCTION / INTENDED PROVIDER
    `sentence-transformers`, model `BAAI/bge-small-en-v1.5`
    (384-dim, cosine similarity). This is what `requirements.txt` declares
    and what `get_provider()` tries FIRST, every time. In any environment
    with normal internet access (or a pre-populated
    `~/.cache/huggingface`), this is the provider that loads and the rest
    of this module is dead code.

SANDBOX DEVIATION (read this before assuming a bug)
    This development sandbox's network egress policy allows pypi.org /
    files.pythonhosted.org / github.com (+release assets) but blocks
    huggingface.co outright (verified: direct requests return HTTP 403).
    `sentence-transformers` cannot download BAAI/bge-small-en-v1.5 here,
    so `get_provider()` falls back, in order, to:

      1. spaCy's `en_core_web_md` -- REAL pretrained (GloVe-derived) word
         vectors, mean-pooled per document by spaCy itself
         (`Doc.vector`), 300-dim, cosine similarity. This is genuine
         semantic similarity (captures synonym/paraphrase relationships,
         not just shared substrings) -- verified: "kubernetes rollback
         runbook and on-call escalation" vs. a differently-worded
         paraphrase scores ~0.74 cosine similarity here, vs. ~0.34 for an
         unrelated sentence. The model wheel is fetched from a GitHub
         release asset (an allowed domain), not from Hugging Face.
      2. A tiny deterministic hashing embedding (see `_HashingProvider`)
         as an absolute last resort so the system never hard-fails for
         lack of *any* embedding backend. This is NOT semantic (no
         synonym/paraphrase generalization) and exists purely so
         chunking/indexing/access-control/aggregation code paths remain
         exercisable even with neither of the above installed.

    `EMBEDDING_DIM` is NOT a hardcoded constant -- it's read from
    whichever provider actually loaded (`get_provider().dimension`), and
    the Qdrant collection is created with that size (see
    `app/vectorstore.py`). Swapping providers (e.g. getting real network
    access to Hugging Face later) means a reindex, not a code change --
    `rebuild_index(force=True)` in `app/tools/semantic_search.py` handles
    that.

Similarity metric: cosine, uniformly, regardless of which provider is
active (`app/vectorstore.py` creates the Qdrant collection with
`Distance.COSINE`).
"""
import hashlib
import logging
from typing import List, Optional, Protocol

logger = logging.getLogger("file_discovery_agent.embeddings")

PREFERRED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
FALLBACK_SPACY_MODEL = "en_core_web_md"
HASHING_FALLBACK_DIM = 384


class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class _SentenceTransformerProvider:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # local import: optional dep

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        embs = self._model.encode(list(texts), normalize_embeddings=True)
        return [e.tolist() for e in embs]


class _SpacyProvider:
    def __init__(self, model_name: str = FALLBACK_SPACY_MODEL):
        import spacy  # local import: optional dep

        self._nlp = spacy.load(model_name, exclude=["parser", "ner", "lemmatizer"])
        self.name = f"spacy:{model_name} (sandbox fallback -- see module docstring)"
        self.dimension = len(self._nlp("dimension probe").vector)

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for doc in self._nlp.pipe([t or " " for t in texts]):
            vec = doc.vector
            norm = float((vec**2).sum() ** 0.5)
            out.append((vec / norm if norm > 0 else vec).tolist())
        return out


class _HashingProvider:
    """Deterministic, dependency-free, NON-semantic fallback -- see module
    docstring's SANDBOX DEVIATION section. Only used if neither
    sentence-transformers nor spaCy can be loaded."""

    def __init__(self, dimension: int = HASHING_FALLBACK_DIM):
        self.name = "hashing_fallback (NOT semantic -- see embeddings.py docstring)"
        self.dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        tokens = (text or "").lower().split()
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm > 0 else vec


_provider: Optional[EmbeddingProvider] = None


def get_provider() -> EmbeddingProvider:
    global _provider
    if _provider is not None:
        return _provider

    try:
        _provider = _SentenceTransformerProvider(PREFERRED_MODEL_NAME)
        logger.info("embedding provider: %s", _provider.name)
        return _provider
    except Exception as exc:  # noqa: BLE001 -- any failure means "try the next tier"
        logger.warning("sentence-transformers unavailable (%s); trying spaCy fallback", exc)

    try:
        _provider = _SpacyProvider()
        logger.info("embedding provider: %s", _provider.name)
        return _provider
    except Exception as exc:  # noqa: BLE001
        logger.warning("spaCy fallback unavailable (%s); using hashing fallback", exc)

    _provider = _HashingProvider()
    logger.info("embedding provider: %s", _provider.name)
    return _provider


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_provider().embed(texts)


def get_embedding_dim() -> int:
    return get_provider().dimension


def get_provider_name() -> str:
    return get_provider().name
