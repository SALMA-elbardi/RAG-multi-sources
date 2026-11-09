"""Modèles de données pour le découpage documentaire (chunking) et la validation."""
from typing import Dict, Any, Union, Optional, List
from pydantic import BaseModel, Field, field_validator


# Types primitifs autorisés dans les métadonnées (compatibilité stricte ChromaDB)
MetadataValue = Union[str, int, float, bool]


class ChunkDocument(BaseModel):
    """
    Représentation unitaire d'un chunk de document normalisé prêt pour l'indexation vectorielle.
    Garantit la traçabilité complète de sa provenance, son niveau de sensibilité RBAC,
    son rang séquentiel et des métadonnées plates compatibles avec ChromaDB.
    """
    chunk_id: str = Field(..., min_length=1, description="Identifiant unique et déterministe du chunk")
    content: str = Field(..., min_length=1, description="Contenu textuel du chunk")
    metadata: Dict[str, MetadataValue] = Field(
        default_factory=dict,
        description="Dictionnaire plat de métadonnées primitives (strictement compatible ChromaDB)"
    )

    @field_validator("metadata")
    @classmethod
    def validate_metadata_types_and_safety(cls, v: Dict[str, Any]) -> Dict[str, MetadataValue]:
        """Garantit l'absence d'objets complexes (pas de list ou dict imbriqués)."""
        clean_meta: Dict[str, MetadataValue] = {}
        for key, val in v.items():
            if val is None:
                continue
            if isinstance(val, (str, int, float, bool)):
                clean_meta[key] = val
            else:
                raise ValueError(
                    f"La métadonnée '{key}' a un type non primitif '{type(val).__name__}' "
                    f"incompatible avec ChromaDB."
                )
        return clean_meta

    @property
    def document_id(self) -> str:
        """Identifiant du document source parent dans inventory.json."""
        return str(self.metadata.get("document_id", ""))

    @property
    def source(self) -> str:
        """Chemin relatif normalisé du document source."""
        return str(self.metadata.get("source", ""))

    @property
    def date_version(self) -> str:
        """Version documentaire ou date de publication."""
        return str(self.metadata.get("date_version", ""))

    @property
    def doc_type(self) -> str:
        """Type documentaire : 'pdf', 'markdown', 'faq' ou 'csv'."""
        return str(self.metadata.get("doc_type", ""))

    @property
    def sensitivity_level(self) -> int:
        """Niveau d'accréditation RBAC (1: Public, 2: Interne, 3: Restreint)."""
        val = self.metadata.get("sensitivity_level")
        if val is None:
            raise ValueError(f"Métadonnée obligatoire 'sensitivity_level' manquante sur le chunk {self.chunk_id}")
        level = int(val)
        if level not in (1, 2, 3):
            raise ValueError(f"Niveau de sensibilité {level} invalide sur le chunk {self.chunk_id} (attendu: 1, 2 ou 3).")
        return level

    @property
    def chunk_index(self) -> int:
        """Index séquentiel du chunk au sein du document parent (0-indexé)."""
        return int(self.metadata.get("chunk_index", 0))

    @property
    def total_chunks(self) -> int:
        """Nombre total de chunks issus du document parent."""
        return int(self.metadata.get("total_chunks", 1))

    @property
    def chunking_strategy(self) -> str:
        """Stratégie appliquée : 'fixed' ou 'structural'."""
        return str(self.metadata.get("chunking_strategy", ""))

    def to_langchain_format(self) -> Dict[str, Any]:
        """Exporte le chunk au format standard LangChain Document."""
        export_meta = dict(self.metadata)
        export_meta["chunk_id"] = self.chunk_id
        return {
            "page_content": self.content,
            "metadata": export_meta
        }

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation en dictionnaire standard."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": dict(self.metadata)
        }


class ChunkValidationReport(BaseModel):
    """Rapport qualitatif et statistique de validation des chunks générés."""
    total_chunks_inspected: int = 0
    valid_chunks_count: int = 0
    rejected_chunks_count: int = 0
    rejected_details: List[Dict[str, Any]] = Field(default_factory=list)
    warnings_count: int = 0
    warnings: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Vrai si aucun chunk n'a été rejeté."""
        return self.rejected_chunks_count == 0


class ChunkingStats(BaseModel):
    """Statistiques d'exécution d'une configuration de chunking."""
    strategy_name: str
    total_documents_input: int = 0
    total_chunks_output: int = 0
    avg_chunk_length: float = 0.0
    min_chunk_length: int = 0
    max_chunk_length: int = 0
    short_chunks_count: int = 0
    target_exceeded_count: int = 0
    metadata_preservation_rate: float = 100.0
    chunks_by_sensitivity: Dict[int, int] = Field(default_factory=dict)
    chunks_by_format: Dict[str, int] = Field(default_factory=dict)


class ChunkComparisonReport(BaseModel):
    """Rapport comparatif entre les différentes stratégies de chunking."""
    configs: Dict[str, ChunkingStats] = Field(default_factory=dict)
    sample_chunks: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)

