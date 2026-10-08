"""
LLM provider abstraction -- Phase 5A (architecture_final.md §9: LLM usage
is "restricted to four narrow, structured roles... never the default
control mechanism", and provider choice is explicitly not an architectural
commitment). The graph (app/graph/build.py) and the goal-interpretation
module (app/llm/goal_interpreter.py) depend only on the `LLMProvider`
protocol below -- never on a specific vendor SDK -- so swapping providers,
or running with none at all, never touches graph/agent logic.

Configuration is entirely environment-based and optional (prompt
requirement 8): with no `DISCOVERY_LLM_PROVIDER` set, `get_default_provider()`
returns `NullProvider`, which never performs any I/O. This is what keeps
the test suite, and any environment without LLM credentials configured,
fully offline -- constructing or calling a provider is never implicitly a
network operation.
"""
import os
from typing import Optional, Protocol, runtime_checkable


class LLMUnavailableError(Exception):
    """Raised by a provider when it cannot produce a completion for any
    reason (not configured, network/timeout failure, SDK missing, API
    error, ...). Callers (app/llm/goal_interpreter.py) treat this
    uniformly as "fall back to deterministic behavior" -- never as
    something to retry indefinitely or surface as a hard failure."""


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface every provider implements. Deliberately just one
    method: providers are a text-in/text-out completion source, nothing
    more -- they never see the tool registry, the database, or
    `user_scope`, and they never execute anything themselves."""

    name: str

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Returns the raw text of the model's response. Implementations
        must raise `LLMUnavailableError` (or let an exception propagate --
        callers catch broadly) rather than returning a partial/placeholder
        string on failure."""
        ...


class NullProvider:
    """The default provider. Always unavailable, and -- critically --
    raises immediately with no I/O of any kind, so using the default
    configuration never requires network access or an API key."""

    name = "none"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMUnavailableError(
            "no LLM provider configured (set DISCOVERY_LLM_PROVIDER to enable one)"
        )


class AnthropicProvider:
    """Thin wrapper over the Anthropic Messages API. Only ever constructed
    when `DISCOVERY_LLM_PROVIDER=anthropic` is explicitly set -- the
    default test run and the default `get_default_provider()` never
    import or instantiate this class, let alone call it."""

    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_tokens: int = 512,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._timeout = timeout
        self._max_tokens = max_tokens
        if not self._api_key:
            raise LLMUnavailableError(
                "DISCOVERY_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set"
            )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import anthropic  # local import: optional dependency, only needed by this provider
        except ImportError as exc:
            raise LLMUnavailableError(f"anthropic package not installed: {exc}") from exc

        try:
            client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout)
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
        except Exception as exc:  # noqa: BLE001 -- any SDK/network/API failure -> safe-fallback signal
            raise LLMUnavailableError(f"anthropic provider call failed: {exc}") from exc


def get_default_provider() -> LLMProvider:
    """Reads `DISCOVERY_LLM_PROVIDER` (default: unset -> "none") and
    constructs the matching provider. Never raises: any construction
    failure (missing API key, missing SDK, unknown provider name) degrades
    to `NullProvider` so a misconfigured environment behaves exactly like
    "no LLM configured" rather than crashing graph construction."""
    name = os.environ.get("DISCOVERY_LLM_PROVIDER", "none").strip().lower()
    if name in ("", "none"):
        return NullProvider()
    if name == "anthropic":
        try:
            return AnthropicProvider(
                model=os.environ.get("DISCOVERY_LLM_MODEL", "claude-haiku-4-5-20251001")
            )
        except Exception:
            return NullProvider()
    return NullProvider()
