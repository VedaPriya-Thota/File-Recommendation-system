"""Phase 10: local-file ingestion feeding the EXISTING BM25 / semantic retrieval. See pipeline.py."""
from app.ingestion.extract import ContentExtractor, ExtractedContent
from app.ingestion.normalize import normalize_text
from app.ingestion.pipeline import IngestionReport, UnknownOwnerError, ingest, ingest_directory
from app.ingestion.source import DiscoveredFile, FileSource, LocalDirectorySource, SourceUnavailableError

__all__ = ["ContentExtractor", "ExtractedContent", "normalize_text", "IngestionReport", "UnknownOwnerError",
           "ingest", "ingest_directory", "DiscoveredFile", "FileSource", "LocalDirectorySource", "SourceUnavailableError"]
