"""
File sources for ingestion. Deliberately tiny so another source could implement the same two methods.

Discovery reads ONLY filesystem metadata (stat), never file contents. Paths are exposed as POSIX paths
RELATIVE to the source root; absolute machine paths never leave this module, and nothing here is ever shown
to an LLM.

Exclusions are fnmatch patterns matched case-insensitively against (a) each directory/file NAME and (b) the
POSIX path relative to the root. A pattern ending in "/" only matches directories ("node_modules/").
"""
import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Protocol, Sequence

DEFAULT_EXCLUSIONS = (
    ".git/", ".svn/", ".hg/", "node_modules/", "__pycache__/", ".venv/", "venv/", "env/", ".idea/", ".vscode/",
    ".cache/", ".pytest_cache/", ".mypy_cache/", ".gradle/", "dist/", "build/", "target/", ".next/", ".tox/",
    "$RECYCLE.BIN/", "System Volume Information/", ".Trash/", ".Trash-*/",
    "*.pyc", "*.class", "*.o", "*.so", "*.dll", "*.exe", "*.tmp", "~$*", ".DS_Store", "Thumbs.db", "desktop.ini",
    "*.min.js", "*.lock", "package-lock.json",
)


class SourceUnavailableError(RuntimeError):
    """The source root is missing/unreadable. The run aborts and deletes NOTHING
    (an unmounted drive must not look like 'every file was deleted')."""


class FileTooLargeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscoveredFile:
    rel_path: str       # POSIX, relative to the root: the only path identity that is stored
    filename: str
    extension: str      # lowercase, no dot; "" if none
    size_bytes: int
    mtime_ns: int
    created_ns: int


class FileSource(Protocol):
    last_errors: int

    def discover(self) -> List[DiscoveredFile]: ...

    def read_bytes(self, rel_path: str, max_bytes: int) -> bytes: ...


def normalize_patterns(patterns: Iterable[str]) -> List[str]:
    out, seen = [], set()
    for p in patterns or []:
        p = (p or "").strip().replace("\\", "/")
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


class Exclusions:
    def __init__(self, patterns: Sequence[str] = ()):
        self.dir_patterns, self.any_patterns = [], []
        for p in normalize_patterns(patterns):
            if p.endswith("/"):
                self.dir_patterns.append(p.rstrip("/").lower())
            else:
                self.any_patterns.append(p.lower())

    @staticmethod
    def _hit(patterns, name, rel):
        for pat in patterns:
            if fnmatch.fnmatchcase(name, pat) or fnmatch.fnmatchcase(rel, pat) or ("/" in pat and rel.startswith(pat.rstrip("*"))):
                return True
        return False

    def dir_excluded(self, name: str, rel: str) -> bool:
        n, r = name.lower(), rel.lower()
        return self._hit(self.dir_patterns, n, r) or self._hit(self.any_patterns, n, r)

    def file_excluded(self, name: str, rel: str) -> bool:
        return self._hit(self.any_patterns, name.lower(), rel.lower())


class LocalDirectorySource:
    """Recursive, deterministic (sorted by relative path) discovery of a local directory.

    * unsupported extensions ARE discovered (metadata-only files)
    * symlinks are skipped, never followed (no escaping the root)
    * `last_errors` counts entries that could not be inspected; the pipeline refuses to delete anything when
      it is non-zero (it cannot be sure a file is really gone)
    * `excluded` counts files/directories skipped by exclusion rules (reported, never silently dropped)
    """

    def __init__(self, root, exclusions: Sequence[str] = (), include_hidden: bool = True, on_progress=None):
        self.root = Path(root).expanduser().resolve()
        self.exclusions = Exclusions(exclusions)
        self.include_hidden = include_hidden
        self.on_progress = on_progress
        self.last_errors = 0
        self.excluded = 0
        self.error_samples: List[str] = []

    def discover(self) -> List[DiscoveredFile]:
        if not self.root.is_dir():
            raise SourceUnavailableError("source root is not an accessible directory")
        errors = 0
        self.excluded = 0
        self.error_samples = []

        def _onerror(exc):
            nonlocal errors
            errors += 1
            if len(self.error_samples) < 20:
                self.error_samples.append(getattr(exc, "filename", None) and os.path.relpath(exc.filename, self.root) or "?")

        found: List[DiscoveredFile] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False, onerror=_onerror):
            rel_dir = Path(dirpath).relative_to(self.root).as_posix()
            rel_dir = "" if rel_dir == "." else rel_dir
            keep = []
            for d in sorted(dirnames):
                rel = f"{rel_dir}/{d}" if rel_dir else d
                if (not self.include_hidden and d.startswith(".")) or self.exclusions.dir_excluded(d, rel):
                    self.excluded += 1
                    continue
                if os.path.islink(os.path.join(dirpath, d)):
                    continue
                keep.append(d)
            dirnames[:] = keep
            for name in filenames:
                rel = f"{rel_dir}/{name}" if rel_dir else name
                if (not self.include_hidden and name.startswith(".")) or self.exclusions.file_excluded(name, rel):
                    self.excluded += 1
                    continue
                path = Path(dirpath) / name
                try:
                    if path.is_symlink():
                        continue
                    st = path.stat()
                except OSError:
                    errors += 1
                    continue
                ext = path.suffix[1:].lower() if path.suffix else ""
                found.append(DiscoveredFile(
                    rel_path=rel, filename=name, extension=ext, size_bytes=st.st_size,
                    mtime_ns=st.st_mtime_ns, created_ns=min(st.st_mtime_ns, st.st_ctime_ns),
                ))
            if self.on_progress:
                self.on_progress(len(found))
        self.last_errors = errors
        found.sort(key=lambda f: f.rel_path)
        return found

    def resolve_inside(self, rel_path: str) -> Path:
        """Absolute path of rel_path, guaranteed to be inside the root (raises PermissionError otherwise)."""
        target = (self.root / rel_path).resolve()
        if self.root != target and self.root not in target.parents:
            raise PermissionError("path escapes source root")
        return target

    def read_bytes(self, rel_path: str, max_bytes: int) -> bytes:
        target = self.resolve_inside(rel_path)
        with open(target, "rb") as fh:
            data = fh.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise FileTooLargeError("file exceeds INGEST_MAX_FILE_BYTES")
        return data
