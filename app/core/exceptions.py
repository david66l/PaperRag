"""Custom exceptions."""


class PaperRagError(Exception):
    """Base exception."""


class DataLoadError(PaperRagError):
    """Failed to load data."""


class EmbeddingError(PaperRagError):
    """Embedding computation failed."""


class IndexError_(PaperRagError):
    """Index build / load failure."""


class RetrievalError(PaperRagError):
    """Retrieval failure."""


class GenerationError(PaperRagError):
    """LLM generation failure."""


class StorageError(PaperRagError):
    """Storage read / write failure."""
