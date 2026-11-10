"""Package de génération RAG avec LLM local (Ollama) et prompt engineering sécurisé."""
from src.generation.models import (
    LLMConfig,
    DocumentCitation,
    RAGResponse,
    LLMClientError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMModelNotFoundError,
    RAGPipelineError,
    extract_citation_from_chunk
)
from src.generation.prompt_builder import (
    PromptBuilder
)
from src.generation.llm_client import (
    OllamaClient
)


def __getattr__(name: str):
    """Import paresseux de RAGPipeline pour éliminer le RuntimeWarning de runpy lors de l'exécution CLI."""
    if name == "RAGPipeline":
        from src.generation.rag_pipeline import RAGPipeline
        return RAGPipeline
    raise AttributeError(f"Le module '{__name__}' n'a pas d'attribut '{name}'")


__all__ = [
    "LLMConfig",
    "DocumentCitation",
    "RAGResponse",
    "LLMClientError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMModelNotFoundError",
    "RAGPipelineError",
    "extract_citation_from_chunk",
    "PromptBuilder",
    "OllamaClient",
    "RAGPipeline"
]

