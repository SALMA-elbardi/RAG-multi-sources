"""Inspecteur pré-inférence de la longueur tokenisée des chunks (politique anti-troncature)."""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable

from src.chunking.models import ChunkDocument
from src.embeddings.embedding_config import EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass
class TokenAuditReport:
    """Rapport d'analyse de la longueur tokenisée des chunks d'un corpus."""
    total_chunks: int = 0
    max_token_limit: int = 507
    min_tokens: int = 0
    max_tokens: int = 0
    avg_tokens: float = 0.0
    tokens_distribution: List[int] = field(default_factory=list)
    overflow_chunks_count: int = 0
    overflow_chunks_details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_overflow(self) -> bool:
        """Vrai si au moins un chunk dépasse le budget maximal autorisé."""
        return self.overflow_chunks_count > 0


class TokenizerInspector:
    """
    Contrôleur de longueur tokenisée prévenant toute troncature silencieuse.
    
    Règles :
      1. Calcule la longueur tokenisée exacte de chaque chunk.
      2. Applique le budget maximal strict (507 tokens = 512 max - 3 préfixe - 2 spéciaux).
      3. Interdit formellement toute troncature silencieuse.
      4. Support d'injection d'une fonction de tokenisation simulée pour les tests unitaires hors ligne.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        tokenize_fn: Optional[Callable[[str], int]] = None
    ):
        self.config = config or EmbeddingConfig.create_structural_config()
        self._tokenizer = None
        self._custom_tokenize_fn = tokenize_fn

    @property
    def tokenizer(self):
        """Chargement paresseux du tokenizer Hugging Face officiel avec son commit SHA-1."""
        if self._custom_tokenize_fn is not None:
            return None

        if self._tokenizer is None:
            from transformers import AutoTokenizer
            logger.info(
                f"Chargement du tokenizer pour '{self.config.model_name}' "
                f"(commit: {self.config.model_revision[:12]})..."
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                revision=self.config.model_revision
            )
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Calcule le nombre de tokens d'une chaîne sans appliquer de troncature."""
        if not text or not text.strip():
            return 0

        if self._custom_tokenize_fn is not None:
            return self._custom_tokenize_fn(text)

        # Utilisation du tokenizer officiel avec truncation=False formel
        tokens = self.tokenizer.encode(text.strip(), truncation=False, add_special_tokens=False)
        return len(tokens)

    def inspect_chunks(self, chunks: List[ChunkDocument]) -> TokenAuditReport:
        """
        Audit exhaustif de la longueur tokenisée de l'ensemble des chunks.
        Retourne un TokenAuditReport complet.
        """
        report = TokenAuditReport(
            total_chunks=len(chunks),
            max_token_limit=self.config.max_content_tokens
        )
        if not chunks:
            return report

        token_counts: List[int] = []

        for c in chunks:
            count = self.count_tokens(c.content)
            token_counts.append(count)

            # Détection des dépassements du budget disponible
            if count > self.config.max_content_tokens:
                report.overflow_chunks_count += 1
                report.overflow_chunks_details.append({
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "doc_type": c.doc_type,
                    "char_length": len(c.content),
                    "token_count": count,
                    "token_limit": self.config.max_content_tokens,
                    "overflow": count - self.config.max_content_tokens,
                    "preview": c.content[:120] + "..." if len(c.content) > 120 else c.content
                })

        report.tokens_distribution = token_counts
        report.min_tokens = min(token_counts)
        report.max_tokens = max(token_counts)
        report.avg_tokens = round(sum(token_counts) / len(token_counts), 1)

        return report

