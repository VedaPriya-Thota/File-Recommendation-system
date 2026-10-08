"""
Qdrant local persistent vector store (Phase 2B) -- architecture_final.md
§6/§10 future semantic layer.

Local persistent mode only: `QdrantClient(path=...)`. No Qdrant server, no
cloud requirement, per the Phase 2B brief. Qdrant's local mode holds a file
lock on the storage directory for as long as a client has it open, so this
module caches ONE client instance per process rather than opening/closing
per call (opening a second client on the same path while one is already
open raises).

Collection vector size is read from the active embedding provider
(`app/embeddings.py`) rather than hardcoded, since which provider is
active can differ by environment (see that module's docstring).
"""
import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import QDRANT_COLLECTION, QDRANT_PATH
from app.embeddings import get_embedding_dim

logger = logging.getLogger("file_discovery_agent.vectorstore")

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
        logger.info("Qdrant local client opened at %s", QDRANT_PATH)
    return _client


def ensure_collection() -> None:
    """Idempotent: creates the collection only if it doesn't already exist.
    If an existing collection's vector size doesn't match the active
    embedding provider's dimension (e.g. the provider changed since the
    collection was built), it is recreated -- a stale-dimension collection
    can't be queried correctly anyway, so silently leaving it around would
    be worse than an explicit rebuild."""
    client = get_client()
    dim = get_embedding_dim()

    if client.collection_exists(QDRANT_COLLECTION):
        info = client.get_collection(QDRANT_COLLECTION)
        existing_dim = info.config.params.vectors.size
        if existing_dim == dim:
            return
        logger.warning(
            "Qdrant collection '%s' has dim=%s but active provider needs dim=%s; recreating (empty)",
            QDRANT_COLLECTION,
            existing_dim,
            dim,
        )
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
