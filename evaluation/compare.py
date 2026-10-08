"""
python -m evaluation.compare [BEFORE.json] [AFTER.json] [OUTPUT_NAME.md]

Renders a before/after comparison of two benchmark result files produced by
`python -m evaluation` (defaults: the committed Phase 7 baseline vs the current
results). Pure post-processing of JSON -- it runs nothing and imports nothing
from `app`.
"""
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CORE = ("A_single_shot_metadata", "B_agentic_deterministic", "C_agentic_personalized", "D_full_system")
RETRIEVAL = ("precision_at_1", "precision_at_3", "recall_at_3", "MRR", "success_rate")
BEHAVIOR = ("clarification_rate", "no_result_rate", "no_result_false_positive_rate", "average_tool_calls",
            "average_iterations", "average_llm_calls", "degraded_rate", "scope_violations")


def _f(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render(before: dict, after: dict) -> str:
    lines = ["# Benchmark comparison (before -> after)", "",
             "Same dataset, variants and thresholds; only production behaviour differs.", "",
             "| variant | metric | before | after |", "|---|---|---|---|"]
    for v in CORE:
        for k in RETRIEVAL:
            lines.append(f"| {v[:1]} | {k} | {_f(before['summary']['overall'][v].get(k))} | {_f(after['summary']['overall'][v].get(k))} |")
        for k in BEHAVIOR:
            lines.append(f"| {v[:1]} | {k} | {_f(before['summary']['behavior'][v].get(k))} | {_f(after['summary']['behavior'][v].get(k))} |")
    ib = {(r["variant"], r["query_id"]): r for r in before["results"]}
    lines += ["", "## Per-query behaviour changes (core variants)", "",
              "| query | variant | before | after | success |", "|---|---|---|---|---|"]
    for r in after["results"]:
        b = ib.get((r["variant"], r["query_id"]))
        if r["variant"] not in CORE or b is None:
            continue
        if (b["status"], b["returned_file_ids"], b["tool_sequence"]) != (r["status"], r["returned_file_ids"], r["tool_sequence"]):
            lines.append(f"| {r['query_id']} | {r['variant'][:1]} | {b['status']} {b['returned_file_ids']} ({b['tool_calls']} tools) "
                         f"| {r['status']} {r['returned_file_ids']} ({r['tool_calls']} tools) | {b['success']} -> {r['success']} |")
    lines += ["", "## Scenarios", ""]
    sb = {s["name"]: s["passed"] for s in before["scenarios"]}
    for s in after["scenarios"]:
        lines.append(f"- {s['name']}: {'PASS' if sb.get(s['name']) else 'FAIL'} -> {'PASS' if s['passed'] else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    b_path = argv[0] if len(argv) > 0 else os.path.join(OUT, "phase7_baseline_results.json")
    a_path = argv[1] if len(argv) > 1 else os.path.join(OUT, "benchmark_results.json")
    with open(b_path) as fb, open(a_path) as fa:
        text = render(json.load(fb), json.load(fa))
    out = os.path.join(OUT, argv[2] if len(argv) > 2 else "phase8_vs_phase7.md")
    with open(out, "w") as f:
        f.write(text)
    print(f"comparison written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
