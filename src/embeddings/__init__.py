"""Package d'embeddings et de stockage vectoriel ChromaDB pour le système RAG."""
from src.embeddings.embedding_config import (
    EmbeddingConfig
)
from src.embeddings.tokenizer_inspector import (
    TokenizerInspector,
    TokenAuditReport
)
from src.embeddings.embedding_validator import (
    EmbeddingValidator
)
from src.embeddings.embedding_manager import (
    EmbeddingManager
)
from src.embeddings.chroma_store import (
    ChromaStore
)


def __getattr__(name: str):
    """Import paresseux des composants de indexation_pipeline pour éliminer le RuntimeWarning lors de l'exécution CLI."""
    if name in ("IndexationPipeline", "IndexingReport"):
        import src.embeddings.indexation_pipeline as ip
        return getattr(ip, name)
    raise AttributeError(f"Le module '{__name__}' n'a pas d'attribut '{name}'")


__all__ = [
    "EmbeddingConfig",
    "TokenizerInspector",
    "TokenAuditReport",
    "EmbeddingValidator",
    "EmbeddingManager",
    "ChromaStore",
    "IndexingReport",
    "IndexationPipeline"
]

