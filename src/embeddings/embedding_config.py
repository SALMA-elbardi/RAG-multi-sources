"""Configuration typée et immuable pour intfloat/multilingual-e5-small et ChromaDB."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Paramètres immuables d'exécution pour la vectorisation sémantique et l'indexation ChromaDB.
    
    Attributs du modèle :
      - model_name : Nom du modèle Hugging Face ('intfloat/multilingual-e5-small').
      - model_revision : Commit SHA-1 immuable officiel du modèle sur Hugging Face.
      - embedding_dim : Dimension géométrique exacte du vecteur (384).
      - max_seq_length : Capacité maximale du tokenizer en tokens (512).
      - max_content_tokens : Budget maximal de tokens alloué au contenu brut (507 = 512 - 3 prefix - 2 spec).
      - document_prefix : Préfixe obligatoire pour les passages/documents ('passage: ').
      - query_prefix : Préfixe obligatoire pour les requêtes utilisateur ('query: ').
      - batch_size : Nombre de chunks par lot d'inférence CPU (valeur initiale recommandée: 16).
      - normalize_embeddings : Normalisation L2 unitaire des vecteurs (True).
      - device : Matériel cible pour le calcul ('cpu' par défaut).
      
    Attributs de stockage ChromaDB :
      - persist_directory : Répertoire persistant local de ChromaDB.
      - collection_name : Nom de la collection vectorielle ciblée.
      - distance_metric : Espace métrique HNSW ('cosine').
      - chunk_file_path : Chemin du fichier JSON source contenant les chunks de Phase 5.
      - chunking_strategy : Stratégie documentaire associée ('structural' ou 'fixed').
    """
    model_name: str = "intfloat/multilingual-e5-small"
    model_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    embedding_dim: int = 384
    max_seq_length: int = 512
    max_content_tokens: int = 507
    document_prefix: str = "passage: "
    query_prefix: str = "query: "
    batch_size: int = 16
    normalize_embeddings: bool = True
    device: str = "cpu"
    
    persist_directory: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "vector_store")))
    collection_name: str = "datanova_corpus_structural"
    distance_metric: str = "cosine"
    chunk_file_path: Path = field(default_factory=lambda: Path("data/processed/chunks_structural.json"))
    chunking_strategy: str = "structural"

    @classmethod
    def create_structural_config(
        cls,
        persist_directory: Optional[Path] = None,
        batch_size: int = 16
    ) -> "EmbeddingConfig":
        """Factory pour l'indexation de référence du corpus structurel (142 chunks)."""
        return cls(
            model_name="intfloat/multilingual-e5-small",
            model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            embedding_dim=384,
            max_seq_length=512,
            max_content_tokens=507,
            document_prefix="passage: ",
            query_prefix="query: ",
            batch_size=batch_size,
            normalize_embeddings=True,
            device="cpu",
            persist_directory=persist_directory or Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "vector_store")),
            collection_name="datanova_corpus_structural",
            distance_metric="cosine",
            chunk_file_path=Path("data/processed/chunks_structural.json"),
            chunking_strategy="structural"
        )

    @classmethod
    def create_fixed_config(
        cls,
        persist_directory: Optional[Path] = None,
        batch_size: int = 16
    ) -> "EmbeddingConfig":
        """Factory pour l'indexation comparative du corpus fixe (195 chunks)."""
        return cls(
            model_name="intfloat/multilingual-e5-small",
            model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            embedding_dim=384,
            max_seq_length=512,
            max_content_tokens=507,
            document_prefix="passage: ",
            query_prefix="query: ",
            batch_size=batch_size,
            normalize_embeddings=True,
            device="cpu",
            persist_directory=persist_directory or Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "vector_store")),
            collection_name="datanova_corpus_fixed",
            distance_metric="cosine",
            chunk_file_path=Path("data/processed/chunks_fixed.json"),
            chunking_strategy="fixed"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation complète de la configuration pour la traçabilité MLOps."""
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.max_seq_length,
            "max_content_tokens": self.max_content_tokens,
            "document_prefix": self.document_prefix,
            "query_prefix": self.query_prefix,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "device": self.device,
            "persist_directory": str(self.persist_directory),
            "collection_name": self.collection_name,
            "distance_metric": self.distance_metric,
            "chunk_file_path": str(self.chunk_file_path),
            "chunking_strategy": self.chunking_strategy
        }
