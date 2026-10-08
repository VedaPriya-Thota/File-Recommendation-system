from app.tools.registry import ToolRegistry
from app.tools import metadata_search as metadata_search_module
from app.tools import bm25_search as bm25_search_module
from app.tools import semantic_search as semantic_search_module
from app.tools import hybrid_search as hybrid_search_module
from app.tools import rerank_candidates as rerank_candidates_module
from app.tools import get_user_activity as get_user_activity_module


def build_registry() -> ToolRegistry:
    """Phase 4A registry: metadata_search + bm25_search + semantic_search +
    hybrid_search + rerank_candidates + get_user_activity.

    Later phases add get_file_details, inspect_file_content here -- same
    `registry.register(Tool(...))` pattern, no changes needed to
    ToolRegistry itself.
    """
    registry = ToolRegistry()
    registry.register(metadata_search_module.make_tool())
    registry.register(bm25_search_module.make_tool())
    registry.register(semantic_search_module.make_tool())
    registry.register(hybrid_search_module.make_tool())
    registry.register(rerank_candidates_module.make_tool())
    registry.register(get_user_activity_module.make_tool())
    return registry
