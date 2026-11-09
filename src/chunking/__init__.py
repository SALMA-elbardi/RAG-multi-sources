"""Package de découpage documentaire (chunking) et de validation pour le système RAG."""
from src.chunking.models import (
    ChunkDocument,
    ChunkValidationReport,
    ChunkingStats,
    ChunkComparisonReport
)
from src.chunking.chunking_config import (
    ChunkingConfig,
    ChunkingStrategy
)
from src.chunking.text_splitter import (
    BaseChunkSplitter,
    FixedTextSplitter,
    StructuralTextSplitter,
    generate_chunk_id
)
from src.chunking.chunk_validator import (
    ChunkValidator
)


def __getattr__(name: str):
    """Import paresseux de ChunkingManager pour éviter le RuntimeWarning de runpy lors de l'exécution CLI."""
    if name == "ChunkingManager":
        from src.chunking.chunking_manager import ChunkingManager
        return ChunkingManager
    raise AttributeError(f"Le module '{__name__}' n'a pas d'attribut '{name}'")


__all__ = [
    "ChunkDocument",
    "ChunkValidationReport",
    "ChunkingStats",
    "ChunkComparisonReport",
    "ChunkingConfig",
    "ChunkingStrategy",
    "BaseChunkSplitter",
    "FixedTextSplitter",
    "StructuralTextSplitter",
    "generate_chunk_id",
    "ChunkValidator",
    "ChunkingManager"
]
