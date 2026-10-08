"""
File sources for ingestion (Phase 10). Deliberately tiny so another source
(e.g. cloud storage -- out of scope) could implement the same two methods.

Discovery reads ONLY filesystem metadata (stat), never file contents. Paths are
exposed as POSIX paths RELATIVE to the source root; absolute machine paths never
leave this module, and nothing here is ever shown to an LLM.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol


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


class LocalDirectorySource:
    """Recursive, deterministic (sorted by relative path) discovery of a local directory.

    * unsupported extensions ARE discovered (metadata-only files)
    * symlinks are skipped, never followed (no escaping the root)
    * `last_errors` counts entries that could not be inspected; the pipeline
      refuses to delete anything when it is non-zero (it cannot be sure a file
      is really gone)
    """

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.last_errors = 0

    def discover(self) -> List[DiscoveredFile]:
        if not self.root.is_dir():
            raise SourceUnavailableError("source root is not an accessible directory")
        errors = 0

        def _onerror(_exc):
            nonlocal errors
            errors += 1

        found: List[DiscoveredFile] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False, onerror=_onerror):
            dirnames.sort()
            for name in filenames:
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
                    rel_path=path.relative_to(self.root).as_posix(),
                    filename=name,
                    extension=ext,
                    size_bytes=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                    created_ns=min(st.st_mtime_ns, st.st_ctime_ns),
                ))
        self.last_errors = errors
        found.sort(key=lambda f: f.rel_path)
        return found

    def read_bytes(self, rel_path: str, max_bytes: int) -> bytes:
        target = (self.root / rel_path).resolve()
        if self.root != target and self.root not in target.parents:
            raise PermissionError("path escapes source root")
        with open(target, "rb") as fh:
            data = fh.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise FileTooLargeError("file exceeds INGEST_MAX_FILE_BYTES")
        return data
