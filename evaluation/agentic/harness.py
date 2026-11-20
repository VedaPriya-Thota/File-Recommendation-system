"""
Evaluation harness for the multi-agent architecture.

Builds a corpus with KNOWN content, connects it, runs each scenario through the real orchestrator (no mocks except an optional scripted
language model), and records what happened: which agents ran, which tools were called, model calls, cache hits, re-plans / rounds,
parallel batches, wall time -- plus the quality signals: did the cited sources contain the required evidence, is the answer supported,
is it plain English.

CLI:  python -m evaluation.agentic.harness           (prints a table; exit code 1 if an expectation fails)
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

OWNER = "local"

FINAL_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Trail Tracker</title>
<meta name="description" content="Log your hikes and see weekly distance"><link rel="stylesheet" href="style.css">
<script src="https://cdn.example.com/chart.min.js"></script></head>
<body><header><nav><a href="index.html">Home</a><a href="stats.html">Stats</a></nav></header>
<h1>Trail Tracker</h1><h2>Log a hike</h2>
<form id="hikeForm" action="/api/hikes" method="post"><input name="trail" placeholder="Trail name"><input name="km" type="number" placeholder="Distance (km)"><button>Save hike</button></form>
<h2>This week</h2><table id="weekTable"><tr><th>Trail</th><th>km</th></tr></table><canvas id="distanceChart"></canvas>
<script>
function saveHike(e){ e.preventDefault(); fetch('/api/hikes', {method:'POST'}); updateChart(); }
function updateChart(){ new Chart(document.getElementById('distanceChart'), {}); }
document.getElementById('hikeForm').addEventListener('submit', saveHike);
</script></body></html>"""

TRAIN_PY = '''"""Training entry point for the churn model."""
import yaml
import torch
from model import build_model
from data import load_data


def main():
    """Read the config, build the model and run training."""
    cfg = yaml.safe_load(open("config.yaml"))
    model = build_model(cfg["hidden_size"])
    data = load_data(cfg["data_path"])
    train(model, data, cfg["epochs"], cfg["learning_rate"])


def train(model, data, epochs, lr):
    """Fit the model with Adam, one pass over the data per epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for x, y in data:
            loss = compute_loss(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        save_checkpoint(model, epoch)


def compute_loss(pred, y):
    """Binary cross-entropy between predictions and labels."""
    return torch.nn.functional.binary_cross_entropy(pred, y)


def save_checkpoint(model, epoch):
    torch.save(model.state_dict(), f"checkpoint_{epoch}.pt")


if __name__ == "__main__":
    main()
'''
MODEL_PY = '''"""Neural network definition."""
import torch.nn as nn


def build_model(hidden_size):
    """A two-layer perceptron for tabular churn data."""
    return nn.Sequential(nn.Linear(12, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1), nn.Sigmoid())
'''
DATA_PY = '''"""Data loading."""
import pandas as pd


def load_data(path):
    """Load the churn CSV and return (features, label) batches."""
    df = pd.read_csv(path)
    return [(row[:-1], row[-1]) for row in df.values]
'''
CONFIG_YAML = "epochs: 20\nlearning_rate: 0.001\nhidden_size: 64\ndata_path: data/churn.csv\nseed: 7\n"
ML_README = """# Churn Predictor

This project predicts which customers are likely to cancel their subscription using a small neural network trained on usage data.

## Usage
Run `python train.py` to train the model.
"""

SALES_CSV_ROWS = {  # month -> revenue (March is the clear peak)
    "2024-01": 41000, "2024-02": 43500, "2024-03": 78250, "2024-04": 52000, "2024-05": 49800, "2024-06": 51200,
}


def build_corpus(root: Path) -> None:
    for d in ("web", "ml/churn", "data", "docs", "plans"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "web" / "final.html").write_text(FINAL_HTML)
    (root / "ml" / "churn" / "train.py").write_text(TRAIN_PY)
    (root / "ml" / "churn" / "model.py").write_text(MODEL_PY)
    (root / "ml" / "churn" / "data.py").write_text(DATA_PY)
    (root / "ml" / "churn" / "config.yaml").write_text(CONFIG_YAML)
    (root / "ml" / "churn" / "README.md").write_text(ML_README)
    rows = ["date,product,sales"]
    for ym, total in SALES_CSV_ROWS.items():
        rows.append(f"{ym}-05,Widget,{total // 2}")
        rows.append(f"{ym}-20,Gadget,{total - total // 2}")
    (root / "data" / "monthly_sales.csv").write_text("\n".join(rows))
    (root / "docs" / "pricing_2023.txt").write_text("Pricing 2023\n\nThe standard plan costs 20 dollars per month. Support is email only.\n")
    (root / "docs" / "pricing_2024.txt").write_text("Pricing 2024\n\nThe standard plan costs 25 dollars per month. Support now includes live chat.\n")
    (root / "docs" / "leadership.txt").write_text("Leadership\n\nThe CEO is Dana Wu. She joined the company in 2019 and leads the executive team.\n")
    (root / "plans" / "quarterly_budget_q3.txt").write_text("Quarterly budget Q3\n\nThe Q3 marketing budget is 30,000 dollars. Travel is capped at 5,000 dollars.\n")
    (root / "plans" / "budget_review_notes.md").write_text("# Budget review\n\nWe reviewed the quarterly budget and agreed to move 2,000 dollars from travel to advertising.\n")
    (root / "plans" / "holiday_party.txt").write_text("Holiday party\n\nThe party is on December 12 at the community hall. Please bring a dessert.\n")


@dataclass
class Scenario:
    id: str
    question: str
    kind: str                                   # expected plan kind (question type)
    agents: List[str]                           # agents that MUST run
    not_agents: List[str] = field(default_factory=list)  # agents that must NOT run (selected dynamically)
    tools: List[str] = field(default_factory=list)       # tools that must have been called
    source: Optional[str] = None                # file that must be among the cited sources (retrieval relevance)
    evidence: Optional[str] = None              # text that must appear inside a cited excerpt
    answer_has: List[str] = field(default_factory=list)
    answer_not: List[str] = field(default_factory=list)
    max_tool_calls: int = 40
    status: str = "ok"
    min_rounds: int = 1


SCENARIOS = [
    Scenario("html_summary", "Summarize final.html", "summary",
             ["Orchestrator", "File Discovery Agent", "Retrieval Agent", "Analysis Agent", "Verification Agent", "Response Agent"],
             tools=["search_files", "extract_html_structure", "verify_claims", "verify_text"], source="final.html", evidence="Trail Tracker",
             answer_has=["Trail Tracker", "form", "hike"], answer_not=["<form", "</", "<script"], max_tool_calls=12),
    Scenario("ml_training", "Analyze my machine learning project and explain how the model is trained.", "project_explain",
             ["Orchestrator", "File Discovery Agent", "Retrieval Agent", "Analysis Agent", "Verification Agent", "Response Agent"],
             tools=["list_project", "read_code_outline", "read_config", "trace_workflow", "verify_claims"], source="train.py", evidence="loss.backward()",
             answer_has=["train", "build_model", "load_data", "epochs", "20"], max_tool_calls=40),
    Scenario("spreadsheet", "Which month has the highest sales?", "data_calc",
             ["Orchestrator", "File Discovery Agent", "Retrieval Agent", "Analysis Agent", "Verification Agent", "Response Agent"],
             tools=["match_tables", "read_table", "analyze_table", "recompute_calc"], source="monthly_sales.csv", answer_has=["March 2024", "78,250"], max_tool_calls=12),
    Scenario("code_explain", "What does the build_model function do?", "code_explain",
             ["Orchestrator", "Retrieval Agent", "Analysis Agent", "Verification Agent", "Response Agent"],
             tools=["search_contents", "verify_claims"], source="model.py", answer_has=["build_model", "model.py"], max_tool_calls=25),
    Scenario("recommend", "Find files about the quarterly budget", "file_search",
             ["Orchestrator", "File Discovery Agent", "Retrieval Agent", "Verification Agent", "Response Agent"], not_agents=["Analysis Agent"],
             tools=["agent_search"], source="quarterly_budget_q3.txt", answer_has=["quarterly_budget_q3.txt"], max_tool_calls=12),
    Scenario("compare", "Compare the pricing 2023 and pricing 2024 documents", "compare",
             ["Orchestrator", "File Discovery Agent", "Retrieval Agent", "Analysis Agent", "Verification Agent", "Response Agent"],
             tools=["compare_documents", "read_section", "verify_claims"], source="pricing_2024.txt", answer_has=["20 dollars", "25 dollars"], max_tool_calls=20),
    Scenario("missing_evidence", "What is the CEO's salary?", "fact",
             ["Orchestrator", "File Discovery Agent", "Retrieval Agent", "Analysis Agent", "Verification Agent", "Response Agent"],
             tools=["check_sufficiency"], answer_has=["couldn't find", "salary"], status="insufficient", min_rounds=2, max_tool_calls=20),
    Scenario("simple_fact", "How many dollars is the Q3 marketing budget?", "fact",
             ["Orchestrator", "Retrieval Agent", "Analysis Agent", "Verification Agent", "Response Agent"], tools=["search_contents", "verify_claims"],
             source="quarterly_budget_q3.txt", evidence="30,000 dollars", answer_has=["30,000"], max_tool_calls=14),
]


def setup(tmp: Optional[Path] = None):
    from app import folders as F
    from app.appconfig import get_settings
    from app.ingestion.pipeline import ingest
    root = Path(tmp or tempfile.mkdtemp(prefix="fds_agentic_"))
    build_corpus(root)
    folder = F.add_folder(OWNER, str(root), label="agentic-eval", scan=False)
    ingest(F.make_source(folder, get_settings()), OWNER, folder_id=folder["folder_id"], index_semantic=True, record_unchanged=False)
    return root, folder


def teardown(folder):
    from app import folders as F
    F.remove_folder(OWNER, folder["folder_id"])


def run_one(sc: Scenario, conversation_id: Optional[str] = None) -> dict:
    from app.assistant.engine import run_assistant
    events: List[dict] = []
    t0 = time.perf_counter()
    res = run_assistant(OWNER, sc.question, events.append, conversation_id=conversation_id)
    res["_events"] = events
    res["_wall"] = round(time.perf_counter() - t0, 3)
    return res


def judge(sc: Scenario, res: dict) -> Dict[str, object]:
    """Evaluate one run against the scenario's expectations. Returns {check: (ok, detail)}."""
    ag = res.get("agents") or {}
    used_agents = set(ag.get("agents", []))
    tools = ag.get("tools", {})
    answer = res.get("answer", "")
    refs = res.get("references", [])
    checks = {}
    checks["plan kind"] = (res.get("task_kind") == sc.kind, res.get("task_kind"))
    checks["status"] = (res.get("status") == sc.status, res.get("status"))
    checks["agents selected"] = (set(sc.agents) <= used_agents, sorted(set(sc.agents) - used_agents))
    checks["agents skipped"] = (not (set(sc.not_agents) & used_agents), sorted(set(sc.not_agents) & used_agents))
    checks["tools used"] = (all(t in tools for t in sc.tools), [t for t in sc.tools if t not in tools])
    checks["tool budget"] = (ag.get("tool_calls", 99) <= sc.max_tool_calls, ag.get("tool_calls"))
    if sc.source:
        checks["relevant source cited"] = (any(r["filename"] == sc.source and not r.get("closest_only") for r in refs), [r["filename"] for r in refs][:4])
    if sc.evidence:
        checks["evidence in excerpt"] = (any(sc.evidence.lower() in e["text"].lower() for r in refs for e in r["excerpts"]), sc.evidence)
    checks["answer content"] = (all(x.lower() in answer.lower() for x in sc.answer_has), [x for x in sc.answer_has if x.lower() not in answer.lower()])
    checks["answer free of raw data"] = (not any(x in answer for x in sc.answer_not) and not re.search(r"\{\s*\"|chunk_id|file_id", answer), None)
    v = res.get("verification", {})
    checks["excerpts verified"] = (v.get("excerpts_failed", 0) == 0, v.get("excerpts_failed"))
    checks["no model calls without a model"] = (ag.get("llm_calls", 0) == 0, ag.get("llm_calls"))
    checks["rounds"] = (ag.get("rounds", 1) >= sc.min_rounds and ag.get("rounds", 1) <= 2, ag.get("rounds"))
    return checks


def summarize(sc: Scenario, res: dict, checks: dict) -> dict:
    ag = res.get("agents") or {}
    return {"scenario": sc.id, "question": sc.question, "kind": res.get("task_kind"), "status": res.get("status"),
            "agents": [a.replace(" Agent", "") for a in ag.get("agents", [])], "tool_calls": ag.get("tool_calls"), "tools": sorted(ag.get("tools", {})),
            "llm_calls": ag.get("llm_calls"), "cache_hits": ag.get("cache_hits"), "rounds": ag.get("rounds"), "parallel_batches": ag.get("parallel_batches"),
            "retries": ag.get("retries"), "seconds": res.get("_wall"), "passed": sum(1 for ok, _ in checks.values() if ok), "checks": len(checks),
            "failed": {k: d for k, (ok, d) in checks.items() if not ok}}


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="fds_agentic_db_")
    os.environ.setdefault("DISCOVERY_DB_TYPE", "sqlite")
    os.environ["DISCOVERY_SQLITE_PATH"] = os.path.join(tmp, "eval.db")
    os.environ["DISCOVERY_QDRANT_PATH"] = os.path.join(tmp, "qdrant")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.db.init_db import ensure_app_schema, init_db
    init_db(with_seed=False)
    ensure_app_schema()
    root, folder = setup()
    rows, failed = [], 0
    try:
        for sc in SCENARIOS:
            res = run_one(sc)
            checks = judge(sc, res)
            row = summarize(sc, res, checks)
            rows.append(row)
            failed += 1 if row["failed"] else 0
    finally:
        teardown(folder)
    hdr = f"{'scenario':<18}{'kind':<16}{'agents':>7}{'tools':>7}{'llm':>5}{'cache':>6}{'rounds':>7}{'par':>5}{'sec':>7}  checks"
    print(hdr)
    for r in rows:
        print(f"{r['scenario']:<18}{r['kind']:<16}{len(r['agents']):>7}{r['tool_calls']:>7}{r['llm_calls']:>5}{r['cache_hits']:>6}{r['rounds']:>7}{r['parallel_batches']:>5}{r['seconds']:>7}  {r['passed']}/{r['checks']}"
              + (f"  FAILED: {json.dumps(r['failed'], default=str)[:200]}" if r["failed"] else ""))
    print(f"\n{len(rows) - failed}/{len(rows)} scenarios met every expectation")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
