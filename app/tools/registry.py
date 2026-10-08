"""
Minimal tool registry.

Deliberately simple (architecture_final.md, prompt §2: "keep the interface
simple, avoid overengineering a generic plugin framework"). A tool is just
a name plus a callable with a fixed signature:

    fn(args: dict, user_scope: UserScope) -> ToolResult

Later tools (bm25_search, semantic_search, hybrid_search, rerank_candidates,
get_file_details, inspect_file_content, get_user_activity) register the
same way -- nothing here is metadata_search-specific.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from app.db.connection import DatabaseError
from app.scope import UserScope


@dataclass
class ToolResult:
    """Structured result every tool returns, regardless of which tool it is.

    `candidates` entries carry enough info for candidate_pool accumulation
    (architecture_final.md §4/§6): file_id, metadata, and a per-source score.
    """
    tool: str
    candidates: List[dict] = field(default_factory=list)
    error: str = None  # set on tool failure; candidates will be empty
    degraded: bool = False


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[dict, UserScope], ToolResult]


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def execute(self, name: str, args: dict, user_scope: UserScope) -> ToolResult:
        tool = self.get(name)
        try:
            return tool.fn(args, user_scope)
        except DatabaseError:
            # architecture_final.md §7/§14: Postgres errors are never
            # degraded around -- access control depends on the DB, so this
            # must propagate as a hard failure, not a soft tool error that
            # the loop tries to replan its way around.
            raise
        except Exception as exc:  # noqa: BLE001 -- non-DB tool errors are soft
            return ToolResult(tool=name, candidates=[], error=str(exc))
