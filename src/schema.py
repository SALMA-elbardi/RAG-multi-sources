"""Schémas de données stricts pour le système RAG et la gouvernance documentaire."""
from enum import IntEnum
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class SensitivityLevel(IntEnum):
    """Niveaux de sensibilité documentaire pour le contrôle d'accès RBAC."""
    PUBLIC = 1        # Accessible à tous les utilisateurs
    INTERNAL = 2      # Réservé aux collaborateurs internes
    RESTRICTED = 3    # Confidentiel / Strictement restreint


DocType = Literal["pdf", "markdown", "faq"]


class ChunkMetadata(BaseModel):
    """
    Métadonnées obligatoires garantissant la traçabilité et la sécurité de chaque chunk.
    Cahier des charges : identifiant, source, date/version, type documentaire et niveau de sensibilité.
    """
    id: str = Field(..., description="Identifiant unique et déterministe du chunk")
    source: str = Field(..., description="Nom ou chemin du document d'origine")
    date_version: str = Field(..., description="Date de publication ou version du document")
    doc_type: DocType = Field(..., description="Type de document : pdf, markdown ou faq")
    sensitivity_level: int = Field(
        ...,
        ge=1,
        le=3,
        description="Niveau de sensibilité RBAC (1: Public, 2: Interne, 3: Restreint)"
    )

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convertit les métadonnées en format compatible ChromaDB (types primitifs)."""
        return {
            "id": self.id,
            "source": self.source,
            "date_version": self.date_version,
            "doc_type": self.doc_type,
            "sensitivity_level": int(self.sensitivity_level)
        }


class DocumentChunk(BaseModel):
    """Représentation d'un chunk prêt pour l'indexation vectorielle."""
    content: str = Field(..., min_length=1, description="Contenu textuel du chunk")
    metadata: ChunkMetadata

