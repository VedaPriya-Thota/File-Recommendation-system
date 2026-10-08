"""
Central configuration for Phase 1.

All values are plain env-var driven with sane local defaults so the project
runs out of the box against the local Postgres instance set up for this
environment. Nothing here is agent logic -- just connection/tunable knobs,
consistent with architecture_final.md §19 ("config items, tunable via
evaluation, not blocking").
"""
import os


def _int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val else default


def _float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val else default


# --- Database ---------------------------------------------------------
DB_HOST = os.environ.get("DISCOVERY_DB_HOST", "localhost")
DB_PORT = _int("DISCOVERY_DB_PORT", 5432)
DB_NAME = os.environ.get("DISCOVERY_DB_NAME", "file_discovery")
DB_USER = os.environ.get("DISCOVERY_DB_USER", "app_user")
DB_PASSWORD = os.environ.get("DISCOVERY_DB_PASSWORD", "app_password")

DSN = (
    f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
    f"user={DB_USER} password={DB_PASSWORD}"
)

# --- Agent loop tunables (architecture_final.md §19) -------------------
MAX_ITERATIONS = _int("DISCOVERY_MAX_ITERATIONS", 6)

# Reflection thresholds (Tier 1, deterministic — §7)
TOO_MANY_CANDIDATES_THRESHOLD = _int("DISCOVERY_TOO_MANY_THRESHOLD", 6)
DOMINANT_SCORE_GAP = _float("DISCOVERY_DOMINANT_SCORE_GAP", 0.25)

# understand_goal ambiguity heuristic thresholds
AMBIGUITY_CLARIFY_THRESHOLD = _float("DISCOVERY_AMBIGUITY_CLARIFY_THRESHOLD", 0.65)

# --- metadata_search short-token matching (Phase 8) ---------------------------
# Tokens this short (e.g. "ai", "hr", "q3") carry too little evidence for plain
# substring matching ("ai" is inside "campaign", "hr" inside "three"), so
# metadata_search matches them only as WHOLE tokens (delimited by
# non-alphanumerics, e.g. the underscores/spaces/dots in filenames). Longer
# tokens keep ordinary case-insensitive substring matching so partial words
# ("budget" -> "budgets") still work. See app/tools/metadata_search.py.
METADATA_SHORT_TOKEN_MAX_LEN = _int("DISCOVERY_METADATA_SHORT_TOKEN_MAX_LEN", 3)

# --- Local-file ingestion (Phase 10) ---------------------------------------------
# Environment-driven; no machine-specific defaults. Empty root/owner means "must
# be supplied" (CLI flags or env) -- ingestion never guesses a directory.
INGEST_ROOT = os.environ.get("DISCOVERY_INGEST_ROOT", "")
INGEST_OWNER_ID = os.environ.get("DISCOVERY_INGEST_OWNER_ID", "")
# Files larger than this are recorded as failed("too_large") instead of being read into memory.
INGEST_MAX_FILE_BYTES = _int("DISCOVERY_INGEST_MAX_FILE_BYTES", 25 * 1024 * 1024)

# --- Semantic layer (Phase 2B) -----------------------------------------
# Local-persistent-mode Qdrant (qdrant_client.QdrantClient(path=...)) --
# no server, no cloud. See app/vectorstore.py / app/tools/semantic_search.py.
QDRANT_PATH = os.environ.get(
    "DISCOVERY_QDRANT_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qdrant_storage"),
)
QDRANT_COLLECTION = os.environ.get("DISCOVERY_QDRANT_COLLECTION", "file_chunks")
SEMANTIC_CHUNK_WORDS = _int("DISCOVERY_SEMANTIC_CHUNK_WORDS", 120)
SEMANTIC_CHUNK_QUERY_LIMIT = _int("DISCOVERY_SEMANTIC_CHUNK_QUERY_LIMIT", 50)

# --- Final ranking policy (Phase 4B; architecture_final.md §10) ---------
# FinalScore = ALPHA * S_retrieval + GAMMA * S_affinity (see
# app/final_ranking.py for the full derivation). Retrieval-dominant with a
# meaningful-but-secondary personalization weight -- architecture_final.md
# §10's "HIGH-equivalent ~ retrieval-dominant with meaningful affinity
# weight" -- not a magic ML-tuned number; a documented deterministic
# default, tunable per §19.
FINAL_RANKING_ALPHA = _float("DISCOVERY_FINAL_RANKING_ALPHA", 0.85)
FINAL_RANKING_GAMMA = _float("DISCOVERY_FINAL_RANKING_GAMMA", 0.15)

# --- LLM provider (Phase 5A; architecture_final.md §9) -------------------
# Entirely optional / environment-based (prompt requirement 8): unset ->
# "none" -> app.llm.provider.NullProvider, which performs no I/O at all.
# The test suite never sets these, so it never needs network access or an
# API key. See app/llm/provider.py for how this is consumed.
LLM_PROVIDER = os.environ.get("DISCOVERY_LLM_PROVIDER", "none")
LLM_MODEL = os.environ.get("DISCOVERY_LLM_MODEL", "claude-haiku-4-5-20251001")

# --- Hybrid retrieval / RRF (Phase 2C) -----------------------------------
# Standard Reciprocal Rank Fusion: RRF(d) = sum_i 1/(k + rank_i(d)).
# k=60 is the constant from Cormack et al. 2009 -- a conventional default,
# not tuned here. See app/tools/hybrid_search.py.
RRF_K = _int("DISCOVERY_RRF_K", 60)
# How many results each sub-tool (bm25_search/semantic_search) is asked
# for before fusion -- higher than the final hybrid top_k so RRF has a
# real ranked list to draw from, not just the eventual winners.
HYBRID_SUBQUERY_TOP_K = _int("DISCOVERY_HYBRID_SUBQUERY_TOP_K", 20)

# --- LLM candidate comparison (Phase 5C; architecture_final.md §6/§9) -----
# Comparison is a tie-breaker over an ALREADY-RANKED, already-evidenced set
# (see app/llm/candidate_comparer.py). It is only invoked when at least two
# of the top COMPARISON_TOP_N deterministic candidates are within
# DOMINANT_SCORE_GAP (the same "not dominant" threshold reflect() uses) of
# the top final_score. COMPARISON_TOP_N hard-bounds how many candidates the
# LLM ever sees; COMPARISON_MIN_CONFIDENCE is the floor below which an LLM
# preference is ignored and the deterministic order stands.
COMPARISON_TOP_N = _int("DISCOVERY_COMPARISON_TOP_N", 3)
COMPARISON_MIN_CONFIDENCE = _float("DISCOVERY_COMPARISON_MIN_CONFIDENCE", 0.5)
