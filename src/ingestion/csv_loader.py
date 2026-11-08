"""Loader spécialisé pour l'ingestion du catalogue de datasets au format CSV."""
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.ingestion.models import NormalizedDocument
from src.ingestion.metadata_manager import MetadataManager

logger = logging.getLogger(__name__)


class CSVLoader:
    """
    Chargeur du catalogue de datasets CSV pour le corpus DataNova Technologies.
    Règles strictes :
      - Traite 'dataset_catalog.csv' encodé en UTF-8.
      - Transforme chaque ligne de dataset en un document unitaire indépendant.
      - Préserve l'intégralité des 12 colonnes dans le texte normalisé ET dans les métadonnées.
      - Formule le contenu textuel de façon riche et sémantique pour la recherche vectorielle.
      - Distingue l'identifiant parent 'document_id' (DOC-CAT-001) de l'ID unitaire 'id' (ex: DS-001).
    """

    EXPECTED_COLUMNS = [
        "dataset_id", "dataset_name", "description", "domain", "owner_team",
        "storage_layer", "storage_format", "sensitivity_level", "update_frequency",
        "retention_period", "quality_rules", "access_policy"
    ]

    def __init__(self, metadata_manager: MetadataManager):
        self.metadata_manager = metadata_manager

    def load(self, file_path: Path) -> List[NormalizedDocument]:
        """Charge le catalogue CSV et retourne la liste des documents unitaires normalisés."""
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier CSV introuvable : {file_path}")

        inv = self.metadata_manager.get_by_path(file_path)
        parent_doc_id = str(inv["id"]) if inv else "DOC-CAT-001"
        parent_version = str(inv.get("version", "2.3.0")) if inv else "2.3.0"
        parent_title = str(inv.get("title", "Catalogue des Datasets")) if inv else "Catalogue des Datasets"

        source_path = file_path.as_posix()
        if "data/raw/" in source_path:
            source_path = "data/raw/" + source_path.split("data/raw/")[-1]

        documents: List[NormalizedDocument] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                # Vérification de l'intégrité des colonnes
                fieldnames = reader.fieldnames or []
                missing_cols = [c for c in self.EXPECTED_COLUMNS if c not in fieldnames]
                if missing_cols:
                    raise ValueError(f"Colonnes obligatoires manquantes dans '{file_path}' : {missing_cols}")

                for idx, row in enumerate(reader, start=1):
                    ds_id = str(row.get("dataset_id", "")).strip()
                    name = str(row.get("dataset_name", "")).strip()
                    desc = str(row.get("description", "")).strip()
                    domain = str(row.get("domain", "")).strip()
                    team = str(row.get("owner_team", "")).strip()
                    layer = str(row.get("storage_layer", "")).strip()
                    fmt = str(row.get("storage_format", "")).strip()
                    raw_sens = row.get("sensitivity_level")
                    freq = str(row.get("update_frequency", "")).strip()
                    ret = str(row.get("retention_period", "")).strip()
                    rules = str(row.get("quality_rules", "")).strip()
                    policy = str(row.get("access_policy", "")).strip()

                    if not ds_id or not name:
                        logger.warning(f"Ligne {idx} du CSV ignorée : dataset_id ou nom vide.")
                        continue

                    try:
                        sensitivity = int(raw_sens) if raw_sens is not None else 1
                        if sensitivity not in (1, 2, 3):
                            raise ValueError()
                    except Exception:
                        raise ValueError(f"Niveau de sensibilité invalide '{raw_sens}' sur le dataset {ds_id} (attendu: 1, 2 ou 3).")

                    # Synthèse textuelle sémantique complète intégrant les 12 dimensions
                    content = (
                        f"Dataset : {name} (Identifiant : {ds_id})\n"
                        f"Description : {desc}\n"
                        f"Domaine d'activité : {domain} | Équipe responsable : {team}\n"
                        f"Stockage Lakehouse : Couche {layer} (Format de stockage : {fmt})\n"
                        f"Fréquence de mise à jour : {freq} | Rétention des données : {ret}\n"
                        f"Règles et contrôles qualité : {rules}\n"
                        f"Politique et habilitations d'accès : {policy} (Niveau de sensibilité : {sensitivity})"
                    )

                    metadata: Dict[str, Any] = {
                        "document_id": parent_doc_id,
                        "dataset_id": ds_id,
                        "dataset_name": name,
                        "domain": domain,
                        "owner_team": team,
                        "storage_layer": layer,
                        "storage_format": fmt,
                        "sensitivity_level": sensitivity,
                        "update_frequency": freq,
                        "retention_period": ret,
                        "access_policy": policy,
                        "source": source_path,
                        "date_version": parent_version,
                        "doc_type": "csv",
                        "title": parent_title
                    }

                    documents.append(
                        NormalizedDocument(
                            id=ds_id,
                            page_content=content,
                            metadata=metadata
                        )
                    )

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Erreur lors de la lecture du fichier CSV '{file_path}' : {e}")

        if len(documents) == 0:
            raise ValueError(f"Le fichier CSV '{file_path}' est vide ou ne contient aucun dataset valide.")

        return documents

