"""python -m evaluation.content [--output PATH] -- run the Phase 11 real-content evaluation (offline, deterministic)."""
import argparse
import os
import sys


def main(argv=None) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ap = argparse.ArgumentParser(prog="python -m evaluation.content")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    from pathlib import Path

    from evaluation.content.report import render_markdown
    from evaluation.content.runner import OUT_DIR, run_content_eval, write_results

    out = run_content_eval()
    base = Path(args.output) if args.output else OUT_DIR / "content_eval_results.json"
    write_results(out, base)
    base.with_suffix(".md").write_text(render_markdown(out))
    ov = out["summary"]["overall"]
    print(f"{len(out['results'])} queries -> {base}\nretrieval: {ov['retrieval']}\nattribution: {ov['attribution']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
