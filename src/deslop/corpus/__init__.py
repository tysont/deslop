# ABOUTME: Corpus package: manifest handling, ingest, normalization, dedupe.
# ABOUTME: Re-exports the Corpus class used by the public API and CLI.
from deslop.corpus.manifest import Corpus, CorpusStats, DocumentEntry, DocumentText, DuplicateError

__all__ = ["Corpus", "CorpusStats", "DocumentEntry", "DocumentText", "DuplicateError"]
