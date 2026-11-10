"""Modèles de données, configurations et structures de citations pour la génération RAG."""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.retrieval.models import RetrievedChunk


class LLMClientError(Exception):
    """Exception générique du client LLM."""
    pass


class LLMConnectionError(LLMClientError):
    """Exception levée en cas d'impossibilité de joindre le serveur local Ollama."""
    pass


class LLMTimeoutError(LLMClientError):
    """Exception levée lors d'un dépassement de délai d'attente de génération."""
    pass


class LLMModelNotFoundError(LLMClientError):
    """Exception levée lorsque le modèle demandé n'est pas installé dans Ollama."""
    pass


class RAGPipelineError(Exception):
    """Exception levée lors d'une défaillance du pipeline d'orchestration RAG."""
    pass


def _resolve_timeout_seconds() -> float:
    """
    Résout et valide le délai d'attente (timeout) en secondes.
    Lit la variable d'environnement OLLAMA_TIMEOUT_SECONDS ou applique 120.0s par défaut.
    Garantit que la valeur est strictement positive.
    """
    raw_val = os.getenv("OLLAMA_TIMEOUT_SECONDS")
    if raw_val is None or not raw_val.strip():
        return 120.0
    try:
        parsed = float(raw_val.strip())
        if parsed <= 0:
            return 120.0
        return parsed
    except (ValueError, TypeError):
        return 120.0


@dataclass(frozen=True)
class LLMConfig:
    """
    Configuration d'exécution pour le modèle de langage local Ollama (Llama 3.2).
    
    Paramètres optimisés pour une machine CPU (16 Go RAM) sans accélération GPU :
      - model_name : Identifiant du modèle Ollama ('llama3.2').
      - base_url : Point d'accès HTTP du démon Ollama ('http://localhost:11434').
      - temperature : Température basse (0.1) pour maximiser le déterminisme et l'ancrage factuel.
      - top_p : Décodage par échantillonnage de noyau (0.9).
      - max_tokens : Budget maximal de tokens générés par réponse (512).
      - timeout_seconds : Délai d'attente réseau maximal (120.0 secondes sur CPU, configurable via OLLAMA_TIMEOUT_SECONDS).
      - num_ctx : Fenêtre de contexte allouée en RAM (4096 tokens).
      - seed : Graine aléatoire déterministe pour la reproductibilité (42).
    """
    model_name: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 512
    timeout_seconds: float = field(default_factory=_resolve_timeout_seconds)
    num_ctx: int = 4096
    seed: Optional[int] = 42

    def __post_init__(self):
        # Validation stricte du timeout_seconds
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ValueError(f"Le délai d'attente (timeout_seconds) doit être un nombre strictement positif (reçu {self.timeout_seconds}).")

        # Validation de max_tokens
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError(f"Le budget maximal de tokens (max_tokens) doit être un entier strictement positif (reçu {self.max_tokens}).")

        # Validation de num_ctx
        if isinstance(self.num_ctx, bool) or not isinstance(self.num_ctx, int) or self.num_ctx <= 0:
            raise ValueError(f"La taille du contexte (num_ctx) doit être un entier strictement positif (reçu {self.num_ctx}).")

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation de la configuration LLM."""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "num_ctx": self.num_ctx,
            "seed": self.seed
        }


@dataclass(frozen=True)
class DocumentCitation:
    """
    Référence certifiée et vérifiée vers un chunk documentaire exploité dans la réponse.
    Construite exclusivement côté backend à partir des métadonnées réelles du Retriever.
    Conserve les champs d'identification et de granularité propres à chaque source.
    """
    document_id: str
    chunk_id: str
    source: str
    doc_type: str
    sensitivity_level: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    faq_id: Optional[str] = None
    dataset_id: Optional[str] = None
    reference: Optional[str] = None
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation pour restitution API."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "doc_type": self.doc_type,
            "sensitivity_level": self.sensitivity_level,
            "page_number": self.page_number,
            "section": self.section,
            "faq_id": self.faq_id,
            "dataset_id": self.dataset_id,
            "reference": self.reference,
            "title": self.title
        }


def extract_citation_from_chunk(chunk: RetrievedChunk) -> DocumentCitation:
    """
    Extrait une DocumentCitation certifiée à partir d'un RetrievedChunk.
    Garantit l'attribution exacte de la page (PDF), de la section (Markdown),
    du dataset_id (CSV) ou de la catégorie (FAQ).
    """
    meta = chunk.metadata or {}
    doc_type = meta.get("doc_type", "inconnu")

    page_number = meta.get("page_number")
    section = meta.get("section")
    faq_id = meta.get("faq_id")
    dataset_id = meta.get("dataset_id")

    reference: Optional[str] = None
    if doc_type == "pdf":
        total = meta.get("total_pages")
        reference = f"Page {page_number}/{total}" if (page_number and total) else (f"Page {page_number}" if page_number else None)
    elif doc_type == "faq":
        cat = meta.get("category")
        reference = f"FAQ ID: {faq_id} (Catégorie: {cat})" if faq_id else None
    elif doc_type == "csv":
        domain = meta.get("domain")
        reference = f"Dataset: {dataset_id} (Domaine: {domain})" if dataset_id else None
    elif doc_type == "markdown":
        reference = f"Section: {section}" if section else None

    return DocumentCitation(
        document_id=str(meta.get("document_id", "inconnu")),
        chunk_id=chunk.chunk_id,
        source=str(meta.get("source", "source_inconnue")),
        doc_type=doc_type,
        sensitivity_level=chunk.sensitivity_level,
        page_number=page_number,
        section=section,
        faq_id=faq_id,
        dataset_id=dataset_id,
        reference=reference,
        title=meta.get("title")
    )


@dataclass(frozen=True)
class RAGResponse:
    """
    Résultat consolidé du pipeline RAG de bout en bout.
    Comprend la réponse générée, les sources certifiées et les métriques de latence.
    """
    query: str
    answer: str
    user_id: str
    clearance_level: int
    sources: List[DocumentCitation]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model_name: str
    has_sufficient_context: bool
    chunks_retrieved_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation complète pour traçabilité MLOps et réponse API."""
        return {
            "query": self.query,
            "answer": self.answer,
            "user_id": self.user_id,
            "clearance_level": self.clearance_level,
            "model_name": self.model_name,
            "has_sufficient_context": self.has_sufficient_context,
            "chunks_retrieved_count": self.chunks_retrieved_count,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "total_time_ms": self.total_time_ms,
            "sources": [s.to_dict() for s in self.sources]
        }
