"""Package de récupération documentaire sémantique (Retriever) et de contrôle d'accès RBAC."""
from src.retrieval.models import (
    ClearanceLevel,
    ROLE_PERMISSIONS,
    InvalidClearanceError,
    InvalidTopKError,
    RBACSecurityError,
    RetrieverError,
    MAX_TOP_K,
    DEFAULT_TOP_K,
    UserContext,
    RetrievedChunk,
    RetrievalResult
)
from src.retrieval.rbac_manager import (
    RBACManager
)


def __getattr__(name: str):
    """Import paresseux de SemanticRetriever pour éliminer le RuntimeWarning de runpy lors de l'exécution CLI."""
    if name == "SemanticRetriever":
        from src.retrieval.semantic_retriever import SemanticRetriever
        return SemanticRetriever
    raise AttributeError(f"Le module '{__name__}' n'a pas d'attribut '{name}'")


__all__ = [
    "ClearanceLevel",
    "ROLE_PERMISSIONS",
    "InvalidClearanceError",
    "InvalidTopKError",
    "RBACSecurityError",
    "RetrieverError",
    "MAX_TOP_K",
    "DEFAULT_TOP_K",
    "UserContext",
    "RetrievedChunk",
    "RetrievalResult",
    "RBACManager",
    "SemanticRetriever"
]

