"""
Agent runtime primitives.

  EventBus   real execution events (which agent, which tool, ok/failed, how long, cache hit) -> streamed to the UI
  Board      the shared blackboard: small structured results are handed between agents (file ids, evidence offsets,
             findings), never whole documents
  Task/Plan  the orchestrator's task graph with dependencies, conditional skipping and bounded retries
  ToolBox    tool registry with PER-AGENT permissions, structured ToolResult, error capture and metrics
  Cache      reusable file-processing results (outlines, HTML structure, table profiles, summaries) keyed by file fingerprint
"""
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

MAX_ATTEMPTS = 2          # per task: one retry
MAX_ROUNDS = 2            # orchestrator re-plans at most once after verification asks for more evidence
MAX_WORKERS = 4


# ---------------------------------------------------------------------------- events
class EventBus:
    """Collects and streams events. Also offers `.step(...)` so older helper code that expects a `Run` keeps working;
    events are attributed to whichever agent/task the CURRENT THREAD is executing."""

    def __init__(self, run_id: str, emit: Callable[[dict], None]):
        self.id, self._emit, self.t0 = run_id, emit, time.perf_counter()
        self.events: List[dict] = []
        self._lock = threading.Lock()
        self.ctx = threading.local()

    def set_context(self, agent: Optional[str], task: Optional[str] = None):
        self.ctx.agent, self.ctx.task = agent, task

    def step(self, kind: str, title: str, detail: str = "", status: str = "done", **extra):
        agent = extra.pop("agent", None) or getattr(self.ctx, "agent", None) or "Orchestrator"
        ev = {"t": round(time.perf_counter() - self.t0, 3), "agent": agent, "task": getattr(self.ctx, "task", None), "kind": kind,
              "title": title, "detail": detail, "status": status, **extra}
        with self._lock:
            self.events.append(ev)
        try:
            self._emit(ev)
        except Exception:  # noqa: BLE001 -- a UI hook must never break the agents
            pass
        return ev


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.c = {"tool_calls": 0, "tool_failures": 0, "cache_hits": 0, "llm_calls": 0, "retries": 0, "rounds": 1, "parallel_batches": 0, "tasks_run": 0, "tasks_skipped": 0}
        self.agents: List[str] = []
        self.tools: Dict[str, int] = {}

    def inc(self, k: str, n: int = 1):
        with self.lock:
            self.c[k] = self.c.get(k, 0) + n

    def agent(self, name: str):
        with self.lock:
            if name not in self.agents:
                self.agents.append(name)

    def tool(self, name: str):
        with self.lock:
            self.tools[name] = self.tools.get(name, 0) + 1

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.c, "agents": list(self.agents), "tools": dict(self.tools)}


# ---------------------------------------------------------------------------- cache
class FileCache:
    """Small thread-safe LRU for expensive per-file results. Keyed by (namespace, file_id, fingerprint, extra) so a changed file never serves stale data."""

    def __init__(self, size: int = 256):
        self.size, self.d, self.lock = size, OrderedDict(), threading.Lock()

    def get_or_compute(self, key, fn: Callable[[], Any], metrics: Optional[Metrics] = None):
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                if metrics:
                    metrics.inc("cache_hits")
                return self.d[key], True
        val = fn()
        with self.lock:
            self.d[key] = val
            while len(self.d) > self.size:
                self.d.popitem(last=False)
        return val, False


CACHE = FileCache()


# ---------------------------------------------------------------------------- blackboard
@dataclass
class Board:
    owner_id: str
    question: str
    plan_q: Any                                   # understand.QuestionPlan
    settings: dict
    conv_hint: str = ""
    bus: Optional[EventBus] = None
    metrics: Metrics = field(default_factory=Metrics)
    # discovery -> retrieval
    candidates: List[dict] = field(default_factory=list)             # [{file_id, filename, why, score, via}]
    target_ids: List[int] = field(default_factory=list)
    discovery_meta: dict = field(default_factory=dict)
    # retrieval -> analysis
    evidence: Dict[int, list] = field(default_factory=dict)          # need index -> [Ev]
    retrieval_meta: Dict[int, dict] = field(default_factory=dict)
    structures: Dict[int, dict] = field(default_factory=dict)        # file_id -> html / code / table / config structure
    passages_read: Dict[int, list] = field(default_factory=dict)     # file_id -> [Ev] (document reads)
    # analysis -> verification
    results: List[dict] = field(default_factory=list)                # per-need answers: {need, ok, assess, ranked, evs}
    findings: Dict[str, Any] = field(default_factory=dict)           # summaries / comparison / calc / trace / html / project / resume ...
    # verification -> response
    verdict: dict = field(default_factory=dict)
    conflicts: List[dict] = field(default_factory=list)
    info: Dict[int, dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    out: dict = field(default_factory=dict)
    health: dict = field(default_factory=dict)
    rounds: int = 1
    retrieval_hints: dict = field(default_factory=dict)              # verification -> orchestrator: how to retrieve more
    lock: threading.RLock = field(default_factory=threading.RLock)


# ---------------------------------------------------------------------------- tasks
@dataclass
class Task:
    id: str
    agent: str                    # discovery | retrieval | analysis | verification | response
    op: str
    goal: str
    args: dict = field(default_factory=dict)
    deps: List[str] = field(default_factory=list)
    status: str = "pending"       # pending | running | done | failed | skipped
    attempts: int = 0
    error: Optional[str] = None
    summary: str = ""
    optional: bool = False        # a failed optional task does not block its dependants


class Plan:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.order: List[str] = []

    def add(self, t: Task) -> Task:
        self.tasks[t.id] = t
        self.order.append(t.id)
        return t

    def ready(self) -> List[Task]:
        out = []
        for tid in self.order:
            t = self.tasks[tid]
            if t.status != "pending":
                continue
            deps = [self.tasks[d] for d in t.deps if d in self.tasks]
            if all(d.status in ("done", "skipped") or (d.status == "failed" and d.optional) for d in deps):
                out.append(t)
            elif any(d.status == "failed" and not d.optional for d in deps):
                t.status, t.error = "skipped", "a prerequisite task failed"
        return out

    def pending(self) -> bool:
        return any(t.status == "pending" for t in self.tasks.values())

    def describe(self) -> List[dict]:
        return [{"id": t.id, "agent": t.agent, "op": t.op, "goal": t.goal, "deps": t.deps, "status": t.status} for t in (self.tasks[i] for i in self.order)]


# ---------------------------------------------------------------------------- tools
@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""
    cached: bool = False
    ms: float = 0.0


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    agents: Set[str]


class ToolBox:
    def __init__(self, specs: Dict[str, ToolSpec], board: Board):
        self.specs, self.board = specs, board

    def call(self, agent: str, name: str, summarize: Optional[Callable[[Any], str]] = None, **args) -> ToolResult:
        b = self.board
        spec = self.specs.get(name)
        if spec is None or agent not in spec.agents:
            b.bus.step("tool", f"{name}: not permitted", f"{agent} may not use this tool", status="error", agent=agent_label(agent), tool=name)
            return ToolResult(False, error="tool not permitted for this agent")
        t0 = time.perf_counter()
        b.metrics.inc("tool_calls")
        b.metrics.tool(name)
        try:
            data = spec.fn(b, **args)
            ms = (time.perf_counter() - t0) * 1000
            cached = bool(isinstance(data, dict) and data.pop("_cached", False))
            res = ToolResult(True, data, cached=cached, ms=ms)
            detail = summarize(data) if summarize else ""
            b.bus.step("tool", f"Used {name}", detail + (" · cached" if cached else ""), agent=agent_label(agent), tool=name, ms=round(ms, 1), cached=cached,
                       args=_short_args(args))
            return res
        except Exception as exc:  # noqa: BLE001 -- tool failures are data for the agent, not crashes
            ms = (time.perf_counter() - t0) * 1000
            b.metrics.inc("tool_failures")
            b.bus.step("tool", f"{name} failed", f"{type(exc).__name__}: {str(exc)[:140]}", status="error", agent=agent_label(agent), tool=name, ms=round(ms, 1), args=_short_args(args))
            return ToolResult(False, error=f"{type(exc).__name__}: {exc}", ms=ms)


def _short_args(args: dict) -> dict:
    out = {}
    for k, v in args.items():
        if isinstance(v, (list, tuple, set)):
            out[k] = f"[{len(v)} item(s)]" if len(v) > 4 else list(v)
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:80]
        else:
            out[k] = type(v).__name__
    return out


AGENT_LABELS = {"orchestrator": "Orchestrator", "discovery": "File Discovery Agent", "retrieval": "Retrieval Agent", "analysis": "Analysis Agent",
                "verification": "Verification Agent", "response": "Response Agent"}


def agent_label(key: str) -> str:
    return AGENT_LABELS.get(key, key)


# ---------------------------------------------------------------------------- scheduler
def run_plan(plan: Plan, board: Board, agents: Dict[str, Any], toolbox: ToolBox, only: Optional[Set[str]] = None) -> None:
    """Executes every runnable task; tasks whose dependencies are satisfied run IN PARALLEL (threads), others wait.
    Each task is retried at most MAX_ATTEMPTS-1 times; failures are recorded and dependants are skipped (or run if the
    failed task was optional). Terminates because every iteration resolves at least one task."""
    bus = board.bus
    while plan.pending():
        ready = [t for t in plan.ready() if (only is None or t.id in only)]
        if not ready:
            break
        if len(ready) > 1:
            board.metrics.inc("parallel_batches")
            bus.step("handoff", f"Running {len(ready)} independent tasks in parallel", ", ".join(f"{agent_label(t.agent)}: {t.op}" for t in ready), agent="Orchestrator")
        if len(ready) == 1:
            _run_task(ready[0], board, agents, toolbox)
        else:
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(ready))) as pool:
                futs = [pool.submit(_run_task, t, board, agents, toolbox) for t in ready]
                for f in as_completed(futs):
                    f.result()


def _run_task(task: Task, board: Board, agents: Dict[str, Any], toolbox: ToolBox) -> None:
    bus = board.bus
    agent = agents[task.agent]
    while task.attempts < MAX_ATTEMPTS:
        task.attempts += 1
        task.status = "running"
        bus.set_context(agent_label(task.agent), task.id)
        board.metrics.agent(agent_label(task.agent))
        board.metrics.inc("tasks_run")
        bus.step("agent_start", task.goal, "" if task.attempts == 1 else f"retry {task.attempts - 1}", status="running", op=task.op)
        try:
            summary = agent.run(task, board, toolbox)
            task.status, task.summary = "done", summary or ""
            bus.step("agent_done", summary or f"{task.op} complete", "", status="done", op=task.op)
            return
        except AgentFailure as exc:
            task.error = str(exc)
            bus.step("agent_done", f"{task.op} failed", str(exc)[:160], status="error", op=task.op)
            if not exc.retryable:
                break
            board.metrics.inc("retries")
            bus.step("retry", f"Retrying {task.op}", "bounded retry " + f"{task.attempts}/{MAX_ATTEMPTS - 1}", status="warning", agent="Orchestrator")
        except Exception as exc:  # noqa: BLE001
            task.error = f"{type(exc).__name__}: {exc}"
            bus.step("agent_done", f"{task.op} crashed", task.error[:160], status="error", op=task.op)
            board.metrics.inc("retries")
    task.status = "failed"
    bus.set_context(None)


class AgentFailure(Exception):
    def __init__(self, msg: str, retryable: bool = True):
        super().__init__(msg)
        self.retryable = retryable
