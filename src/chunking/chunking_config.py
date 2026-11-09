"""Configurations typées pour les stratégies de découpage documentaire (chunking)."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ChunkingStrategy(str, Enum):
    """Stratégies de chunking disponibles."""
    FIXED = "fixed"
    STRUCTURAL = "structural"


@dataclass
class ChunkingConfig:
    """
    Paramètres configurables d'une stratégie de découpage.
    
    Clarification importante sur la taille cible :
      La taille cible ('chunk_size') est une valeur indicative et optimale de découpage,
      et non une limite absolue rigide. Selon les frontières de phrases, de paragraphes,
      les blocs de code ou les unités structurelles préservées (ex: fiches datasets CSV,
      paires FAQ, ou sous-sections Markdown), certains chunks peuvent dépasser modérément
      cette cible sans être rejetés.
      
    Attributs :
      - strategy : Nom de la stratégie ('fixed' ou 'structural').
      - chunk_size : Taille cible indicative en caractères.
      - chunk_overlap : Recouvrement entre chunks consécutifs.
      - min_chunk_size : Seuil informatif pour signaler un chunk très court.
      - soft_max_size : Seuil informatif de tolérance pour le dépassement de taille cible.
      - separators : Séparateurs hiérarchiques pour RecursiveCharacterTextSplitter.
      - markdown_max_section_size : Seuil au-delà duquel une section Markdown est sous-découpée.
      - preserve_faq_atomicity : Maintient la paire Q/R entière en mode structurel.
      - preserve_csv_atomicity : Maintient la fiche dataset entière en mode structurel.
    """
    strategy: str = ChunkingStrategy.FIXED.value
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 40
    soft_max_size: int = 700
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    
    # Paramètres structurels spécifiques
    markdown_max_section_size: int = 1200
    preserve_faq_atomicity: bool = True
    preserve_csv_atomicity: bool = True

    @classmethod
    def create_fixed(
        cls,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 40,
        soft_max_size: int = 700
    ) -> "ChunkingConfig":
        """Configuration pour le chunking fixe (uniforme)."""
        return cls(
            strategy=ChunkingStrategy.FIXED.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            soft_max_size=soft_max_size,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @classmethod
    def create_structural(
        cls,
        chunk_size: int = 600,
        chunk_overlap: int = 60,
        min_chunk_size: int = 40,
        soft_max_size: int = 900,
        markdown_max_section_size: int = 1200
    ) -> "ChunkingConfig":
        """Configuration pour le chunking structurel (sémantique par format)."""
        return cls(
            strategy=ChunkingStrategy.STRUCTURAL.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            soft_max_size=soft_max_size,
            separators=["\n\n", "\n", ". ", " ", ""],
            markdown_max_section_size=markdown_max_section_size,
            preserve_faq_atomicity=True,
            preserve_csv_atomicity=True
        )
