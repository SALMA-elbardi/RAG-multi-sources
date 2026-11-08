"""Loader spécialisé pour l'ingestion de la FAQ technique au format JSON."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.ingestion.models import NormalizedDocument
from src.ingestion.metadata_manager import MetadataManager

logger = logging.getLogger(__name__)


class JSONLoader:
    """
    Chargeur de FAQ JSON pour le corpus DataNova Technologies.
    Règles strictes :
      - Traite 'technical_faq.json'.
      - Découpe chaque paire question-réponse en un document indépendant unitaire.
      - Le contenu textuel formate explicitement la question et la réponse.
      - Distingue l'identifiant parent 'document_id' (ex: DOC-FAQ-001) de l'ID unitaire 'id' (ex: FAQ-DE-001).
      - Préserve la catégorie, la source, la version et le niveau de sensibilité spécifique.
    """

    def __init__(self, metadata_manager: MetadataManager):
        self.metadata_manager = metadata_manager

    def load(self, file_path: Path) -> List[NormalizedDocument]:
        """Charge le fichier FAQ JSON et retourne la liste des documents unitaires normalisés."""
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier FAQ JSON introuvable : {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as e:
            raise ValueError(f"Impossible de parser le fichier JSON '{file_path}' : {e}")

        if not isinstance(entries, list):
            raise ValueError(f"Le fichier JSON '{file_path}' doit contenir une liste d'objets Q/A.")

        if len(entries) == 0:
            raise ValueError(f"Le fichier JSON '{file_path}' est vide (0 entrée trouvée).")

        # Métadonnées officielles du document parent dans inventory.json
        inv = self.metadata_manager.get_by_path(file_path)
        parent_doc_id = str(inv["id"]) if inv else "DOC-FAQ-001"
        parent_version = str(inv.get("version", "1.5.0")) if inv else "1.5.0"
        parent_title = str(inv.get("title", "FAQ Technique")) if inv else "FAQ Technique"

        source_path = file_path.as_posix()
        if "data/raw/" in source_path:
            source_path = "data/raw/" + source_path.split("data/raw/")[-1]

        documents: List[NormalizedDocument] = []

        for idx, item in enumerate(entries, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"L'entrée n°{idx} de '{file_path}' n'est pas un dictionnaire JSON valide.")

            q_id = str(item.get("id", f"FAQ-{idx}")).strip()
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            category = str(item.get("category", "Général")).strip()
            raw_sens = item.get("sensitivity_level")

            if not question or not answer:
                logger.warning(f"Entrée FAQ {q_id} ignorée : question ou réponse vide.")
                continue

            if raw_sens is None:
                raise ValueError(f"L'entrée FAQ {q_id} ne possède pas de 'sensitivity_level'.")

            try:
                sensitivity = int(raw_sens)
                if sensitivity not in (1, 2, 3):
                    raise ValueError()
            except Exception:
                raise ValueError(f"Niveau de sensibilité invalide '{raw_sens}' sur l'entrée FAQ {q_id} (attendu: 1, 2 ou 3).")

            date_ver = str(item.get("date_version") or parent_version).strip()
            content = f"Question : {question}\nRéponse : {answer}"

            metadata: Dict[str, Any] = {
                "document_id": parent_doc_id,
                "faq_id": q_id,
                "category": category,
                "source": source_path,
                "date_version": date_ver,
                "doc_type": "faq",
                "sensitivity_level": sensitivity,
                "title": parent_title
            }

            documents.append(
                NormalizedDocument(
                    id=q_id,
                    page_content=content,
                    metadata=metadata
                )
            )

        return documents

