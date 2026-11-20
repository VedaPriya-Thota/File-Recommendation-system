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


class OpenAICompatibleProvider:
    """Chat-completions client for any OpenAI-compatible endpoint -- including LOCAL servers such as Ollama
    (http://localhost:11434/v1) or LM Studio, which keep document text on this machine. Pure stdlib (urllib)."""

    name = "openai_compatible"

    def __init__(self, model: str, base_url: str, api_key: Optional[str] = None, timeout: float = 60.0, max_tokens: int = 1200):
        if not model or not base_url:
            raise LLMUnavailableError("openai_compatible provider needs a model and a base URL")
        self._model, self._base = model, base_url.rstrip("/")
        self._key = api_key or os.environ.get("DISCOVERY_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self._timeout, self._max_tokens = timeout, max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import json as _json
        import urllib.request

        body = _json.dumps({
            "model": self._model, "max_tokens": self._max_tokens, "temperature": 0.2,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        try:
            req = urllib.request.Request(self._base + "/chat/completions", data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 -- user-configured endpoint
                data = _json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailableError(f"openai_compatible call failed: {type(exc).__name__}") from exc


_RUNTIME_KEYS: dict = {}  # in-memory only (never written to disk or the database)


def set_runtime_api_key(provider: str, key: str) -> None:
    if key:
        _RUNTIME_KEYS[provider] = key
    else:
        _RUNTIME_KEYS.pop(provider, None)


def has_api_key(provider: str) -> bool:
    env = {"anthropic": "ANTHROPIC_API_KEY", "openai_compatible": "DISCOVERY_LLM_API_KEY"}.get(provider)
    return bool(_RUNTIME_KEYS.get(provider) or (env and os.environ.get(env)) or (provider == "openai_compatible" and os.environ.get("OPENAI_API_KEY")))


def _is_local_url(url: str) -> bool:
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


def llm_status() -> dict:
    """What the application will actually do with the configured model (used by the UI and by every caller).
    Privacy gate: an EXTERNAL provider is only used when the user explicitly consented in Settings."""
    cfg = _stored_model_config()
    provider, model, base = cfg["provider"], cfg["model"], cfg["base_url"]
    external = provider == "anthropic" or (provider == "openai_compatible" and not _is_local_url(base))
    status = {"provider": provider, "model": model, "base_url": base, "external": external,
              "has_key": has_api_key(provider) if provider != "none" else False, "active": False, "reason": ""}
    if provider == "none":
        status["reason"] = "No model configured: deterministic analysis only."
    elif external and not cfg["allow_external"]:
        status["reason"] = "External model selected but sending excerpts to it is not allowed (Settings → Privacy)."
    elif provider == "anthropic" and not status["has_key"]:
        status["reason"] = "ANTHROPIC_API_KEY is not set."
    elif provider == "openai_compatible" and (not model or not base):
        status["reason"] = "Model name and base URL are required."
    else:
        status["active"] = True
    return status


def _stored_model_config() -> dict:
    try:
        from app.appconfig import get_settings
        s = get_settings()
        m, allow = s["model"], s["privacy"]["allow_external_llm"]
        if m["provider"] != "none" or os.environ.get("DISCOVERY_LLM_PROVIDER", "none").lower() in ("", "none"):
            return {**m, "allow_external": allow}
    except Exception:  # noqa: BLE001
        pass
    env_provider = os.environ.get("DISCOVERY_LLM_PROVIDER", "none").strip().lower()  # legacy env-only configuration
    return {"provider": env_provider if env_provider in ("anthropic", "openai_compatible") else "none",
            "model": os.environ.get("DISCOVERY_LLM_MODEL", ""), "base_url": os.environ.get("DISCOVERY_LLM_BASE_URL", ""),
            "timeout_s": 60, "max_tokens": 1200, "allow_external": os.environ.get("DISCOVERY_ALLOW_EXTERNAL_LLM") == "1"}


def get_default_provider() -> LLMProvider:
    """Settings-driven (Settings page) with legacy env fallback. Never raises: any problem -- or a missing
    privacy consent for an external provider -- degrades to NullProvider (no I/O)."""
    try:
        st = llm_status()
        if not st["active"]:
            return NullProvider()
        cfg = _stored_model_config()
        if st["provider"] == "anthropic":
            return AnthropicProvider(model=st["model"] or "claude-haiku-4-5-20251001",
                                     api_key=_RUNTIME_KEYS.get("anthropic"), timeout=float(cfg["timeout_s"]),
                                     max_tokens=int(cfg["max_tokens"]))
        return OpenAICompatibleProvider(model=st["model"], base_url=st["base_url"], api_key=_RUNTIME_KEYS.get("openai_compatible"),
                                        timeout=float(cfg["timeout_s"]), max_tokens=int(cfg["max_tokens"]))
    except Exception:  # noqa: BLE001
        return NullProvider()
