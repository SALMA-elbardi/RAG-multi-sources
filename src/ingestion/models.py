"""Modèles de données communs pour le pipeline d'ingestion documentaire multi-sources."""
from dataclasses import dataclass, field
from typing import Dict, Any, Union, Optional, List
from pydantic import BaseModel, Field, field_validator


# Types primitifs autorisés dans les métadonnées (compatibilité stricte ChromaDB)
MetadataValue = Union[str, int, float, bool]


class NormalizedDocument(BaseModel):
    """
    Structure commune normalisée pour tous les documents ingérés du corpus.
    Distingue rigoureusement :
      - 'id' : identifiant stable unique du document normalisé (ex: 'DOC-SEC-001#p1', 'FAQ-DE-001', 'DS-001').
      - 'document_id' : identifiant du document d'origine dans inventory.json (ex: 'DOC-SEC-001', 'DOC-FAQ-001').
    """
    id: str = Field(..., min_length=1, description="Identifiant unique et stable du document normalisé")
    page_content: str = Field(..., min_length=1, description="Contenu textuel extrait et nettoyé")
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
                    f"La métadonnée '{key}' a un type complexe non primitif '{type(val).__name__}' "
                    f"incompatible avec ChromaDB."
                )
        return clean_meta

    @property
    def document_id(self) -> str:
        """Identifiant du document source parent."""
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
            raise ValueError(f"Métadonnée obligatoire 'sensitivity_level' manquante sur le document {self.id}")
        level = int(val)
        if level not in (1, 2, 3):
            raise ValueError(f"Le niveau de sensibilité {level} du document {self.id} est invalide (doit être 1, 2 ou 3).")
        return level

    def to_langchain_format(self) -> Dict[str, Any]:
        """Exporte le document au format exact attendu par LangChain."""
        # Synchronise 'id' et 'document_id' dans le dictionnaire de métadonnées exporté
        export_meta = dict(self.metadata)
        export_meta["id"] = self.id
        export_meta["document_id"] = self.document_id
        return {
            "page_content": self.page_content,
            "metadata": export_meta
        }


@dataclass
class IngestionReport:
    """Rapport statistique et qualitatif d'exécution du pipeline d'ingestion."""
    total_files_detected: int = 0
    files_by_format: Dict[str, int] = field(default_factory=dict)
    total_documents_loaded: int = 0
    documents_by_format: Dict[str, int] = field(default_factory=dict)
    documents_by_sensitivity: Dict[int, int] = field(default_factory=dict)
    
    # Traçabilité des anomalies et exclusions (pas de perte silencieuse)
    ignored_pages_count: int = 0
    ignored_pages_details: List[Dict[str, Any]] = field(default_factory=list)
    metadata_discrepancies: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rejected_documents_count: int = 0
    errors_count: int = 0
    errors: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files_detected": self.total_files_detected,
            "files_by_format": self.files_by_format,
            "total_documents_loaded": self.total_documents_loaded,
            "documents_by_format": self.documents_by_format,
            "documents_by_sensitivity": self.documents_by_sensitivity,
            "ignored_pages_count": self.ignored_pages_count,
            "ignored_pages_details": self.ignored_pages_details,
            "metadata_discrepancies": self.metadata_discrepancies,
            "anomalies": self.anomalies,
            "warnings": self.warnings,
            "rejected_documents_count": self.rejected_documents_count,
            "errors_count": self.errors_count,
            "errors": self.errors,
            "execution_time_seconds": round(self.execution_time_seconds, 3)
        }
