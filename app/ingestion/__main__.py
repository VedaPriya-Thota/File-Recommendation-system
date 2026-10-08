"""
python -m app.ingestion --root DIR --owner USER_ID [--no-semantic]

Root/owner default to DISCOVERY_INGEST_ROOT / DISCOVERY_INGEST_OWNER_ID; there is
no built-in directory. Prints the structured run report (counts + relative paths
+ short error codes; never document text). Offline: HF access is disabled so an
uncached model falls back immediately instead of attempting a download.
"""
import argparse
import json
import os
import sys


def main(argv=None) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from app import config
    from app.ingestion import UnknownOwnerError, SourceUnavailableError, ingest_directory

    ap = argparse.ArgumentParser(prog="python -m app.ingestion")
    ap.add_argument("--root", default=config.INGEST_ROOT)
    ap.add_argument("--owner", default=config.INGEST_OWNER_ID)
    ap.add_argument("--no-semantic", action="store_true", help="BM25/PostgreSQL only; skip chunks/embeddings/Qdrant")
    args = ap.parse_args(argv)
    if not args.root or not args.owner:
        print("error: --root and --owner (or DISCOVERY_INGEST_ROOT / DISCOVERY_INGEST_OWNER_ID) are required", file=sys.stderr)
        return 2
    try:
        report = ingest_directory(args.root, args.owner, index_semantic=not args.no_semantic)
    except (UnknownOwnerError, SourceUnavailableError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
