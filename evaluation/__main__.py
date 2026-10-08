"""
python -m evaluation [--output-dir DIR] [--init-db] [--latency] [--qdrant-path DIR]

Runs the Phase 7 benchmark against the existing local database and writes
benchmark_results.json + benchmark_report.md. Fully offline: LLM stages use
scripted local providers; HF access is disabled (HF_HUB_OFFLINE=1) so a
missing/uncached model falls back immediately instead of attempting a download.
"""
import argparse
import os
import sys
import tempfile


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation")
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    ap.add_argument("--init-db", action="store_true",
                    help="DESTRUCTIVE: re-apply schema + seed (python -m app.db.init_db) before running")
    ap.add_argument("--latency", action="store_true", help="include environment-specific latency fields")
    ap.add_argument("--qdrant-path", default=None, help="Qdrant local dir (default: fresh temp dir; repo storage untouched)")
    args = ap.parse_args(argv)

    # Must be set BEFORE any app module is imported (config reads env at import time).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DISCOVERY_QDRANT_PATH", args.qdrant_path or tempfile.mkdtemp(prefix="p7_qdrant_"))

    if args.init_db:
        from app.db.init_db import init_db
        init_db(with_seed=True)

    from evaluation.report import render_markdown
    from evaluation.runner import EvaluationSetupError, run_benchmark, write_results

    try:
        output = run_benchmark(include_latency=args.latency)
    except EvaluationSetupError as exc:
        print(f"benchmark setup error: {exc}", file=sys.stderr)
        return 2

    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    md_path = os.path.join(args.output_dir, "benchmark_report.md")
    write_results(output, json_path)
    with open(md_path, "w") as f:
        f.write(render_markdown(output))
    n_fail = sum(1 for s in output["scenarios"] if not s["passed"])
    print(f"{len(output['results'])} records written to {json_path}\nreport: {md_path}\n"
          f"scenarios failing: {n_fail}/{len(output['scenarios'])} (a failing scenario is a finding, not a crash)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
