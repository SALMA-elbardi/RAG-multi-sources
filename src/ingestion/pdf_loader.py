"""Loader spécialisé pour l'extraction page par page des documents PDF actifs."""
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader

from src.ingestion.models import NormalizedDocument
from src.ingestion.metadata_manager import MetadataManager

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Chargeur de documents PDF pour le corpus DataNova Technologies.
    Règles strictes :
      - Extrait le texte page par page.
      - Conserve le numéro de page dans les métadonnées ('page_number', 'total_pages').
      - Distingue l'identifiant parent 'document_id' (ex: DOC-SEC-001) de l'ID unitaire 'id' (DOC-SEC-001#p1).
      - Ne perd jamais silencieusement une page : toute page ignorée est tracée dans 'ignored_pages'.
      - Détecte les fichiers vides, corrompus ou illisibles.
    """

    def __init__(self, metadata_manager: MetadataManager):
        self.metadata_manager = metadata_manager

    def load(self, file_path: Path) -> Tuple[List[NormalizedDocument], List[Dict[str, Any]]]:
        """
        Extrait les pages d'un fichier PDF.
        Retourne (documents_normalises, pages_ignorees_details).
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier PDF introuvable : {file_path}")

        inv = self.metadata_manager.get_by_path(file_path)
        if not inv:
            logger.warning(f"Le PDF {file_path} n'est pas répertorié dans inventory.json.")
            doc_id = file_path.stem
            sensitivity = 2
            version = "1.0.0"
            title = file_path.stem
            owner_team = ""
        else:
            doc_id = str(inv["id"])
            sensitivity = int(inv["sensitivity_level"])
            version = str(inv.get("version", "1.0.0"))
            title = str(inv.get("title", file_path.stem))
            owner_team = str(inv.get("owner_team", ""))

        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            raise ValueError(f"Impossible de lire le fichier PDF corrompu ou illisible '{file_path}' : {e}")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError(f"Le fichier PDF '{file_path}' ne contient aucune page.")

        normalized_docs: List[NormalizedDocument] = []
        ignored_pages: List[Dict[str, Any]] = []

        source_path = file_path.as_posix()
        if "data/raw/" in source_path:
            source_path = "data/raw/" + source_path.split("data/raw/")[-1]

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                text = ""
                logger.error(f"Erreur d'extraction sur la page {page_idx} de {file_path} : {e}")

            clean_text = text.strip()

            # Règle MLOps : si une page contient moins de 10 caractères utiles (page blanche ou non extractible),
            # elle est consignée avec précision pour traçabilité (zéro perte silencieuse).
            if len(clean_text) < 10:
                ignored_pages.append({
                    "file": file_path.name,
                    "source": source_path,
                    "document_id": doc_id,
                    "page_number": page_idx,
                    "total_pages": total_pages,
                    "char_count": len(clean_text),
                    "reason": "page_vide_ou_texte_insuffisant",
                    "preview": clean_text[:50]
                })
                continue

            page_id = f"{doc_id}#p{page_idx}"
            metadata: Dict[str, Any] = {
                "document_id": doc_id,
                "source": source_path,
                "date_version": version,
                "doc_type": "pdf",
                "sensitivity_level": sensitivity,
                "page_number": page_idx,
                "total_pages": total_pages,
                "title": title,
                "owner_team": owner_team
            }

            normalized_docs.append(
                NormalizedDocument(
                    id=page_id,
                    page_content=clean_text,
                    metadata=metadata
                )
            )

        return normalized_docs, ignored_pages

