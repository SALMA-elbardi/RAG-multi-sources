"""Gestionnaire et réconciliateur strict des métadonnées issues d'inventory.json."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MetadataDiscrepancy(BaseModel):
    """Représente une incohérence constatée entre un document et inventory.json."""
    document_id: str
    source_file: str
    field_name: str
    inventory_value: Any
    extracted_value: Any
    description: str


class MetadataManager:
    """
    Gestionnaire officiel des métadonnées du corpus actif.
    Règles strictes :
      - Charge et indexe 'inventory.json'.
      - Permet la recherche par chemin relatif ou par identifiant.
      - Détecte et signale toute divergence ou métadonnée manquante.
      - N'invente et ne corrige JAMAIS silencieusement une valeur.
    """

    MANDATORY_INVENTORY_FIELDS = ["id", "path", "title", "type", "version", "sensitivity_level", "owner_team", "is_active"]

    def __init__(self, inventory_path: Optional[Path] = None):
        self.inventory_path = inventory_path or Path("data/raw/inventory.json")
        self._by_path: Dict[str, Dict[str, Any]] = {}
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self.discrepancies: List[MetadataDiscrepancy] = []
        self._load_inventory()

    def _normalize_key(self, path_str: str) -> str:
        """Normalise un chemin de fichier en notation standard Unix minuscule."""
        p = Path(path_str).as_posix().lower()
        if "data/raw/" in p:
            p = p.split("data/raw/")[-1]
        elif "/raw/" in p:
            p = p.split("/raw/")[-1]
        return p.lstrip("./")

    def _load_inventory(self) -> None:
        """Charge, valide et indexe les documents actifs d'inventory.json."""
        if not self.inventory_path.exists():
            raise FileNotFoundError(f"Le fichier d'inventaire officiel est introuvable : {self.inventory_path}")

        with open(self.inventory_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        if not isinstance(entries, list):
            raise ValueError("Le fichier inventory.json doit contenir une liste d'objets de métadonnées.")

        for idx, item in enumerate(entries):
            # Contrôle de la complétude de l'inventaire
            missing = [f for f in self.MANDATORY_INVENTORY_FIELDS if f not in item]
            if missing:
                raise ValueError(f"Entrée n°{idx} d'inventory.json invalide : champs obligatoires manquants {missing}")

            doc_id = str(item["id"])
            doc_path = str(item["path"])
            norm_path = self._normalize_key(doc_path)

            # Validation du niveau de sensibilité de l'inventaire
            sens = item.get("sensitivity_level")
            if not isinstance(sens, int) or sens not in (1, 2, 3):
                raise ValueError(
                    f"Le document '{doc_id}' dans inventory.json a un niveau de sensibilité invalide : {sens}. "
                    f"Seules les valeurs entières 1, 2 ou 3 sont autorisées."
                )

            self._by_id[doc_id] = item
            self._by_path[norm_path] = item

    def get_by_path(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Recherche la fiche d'inventaire officielle correspondant à un chemin de fichier."""
        key = self._normalize_key(str(file_path))
        return self._by_path.get(key)

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Recherche la fiche d'inventaire officielle à partir de son document_id."""
        return self._by_id.get(str(doc_id))

    def get_active_documents(self) -> List[Dict[str, Any]]:
        """Retourne la liste de tous les documents marqués 'is_active: true'."""
        return [doc for doc in self._by_id.values() if doc.get("is_active") is True]

    def detect_discrepancies(
        self,
        file_path: Path,
        extracted_metadata: Dict[str, Any]
    ) -> List[MetadataDiscrepancy]:
        """
        Compare les métadonnées extraites d'un fichier source avec celles déclarées dans inventory.json.
        Signale toute incohérence sans écraser silencieusement.
        """
        found_discrepancies: List[MetadataDiscrepancy] = []
        inv = self.get_by_path(file_path)

        if not inv:
            disc = MetadataDiscrepancy(
                document_id=str(extracted_metadata.get("id") or extracted_metadata.get("document_id") or "INCONNU"),
                source_file=file_path.as_posix(),
                field_name="inventory_entry",
                inventory_value=None,
                extracted_value=extracted_metadata,
                description=f"Le fichier '{file_path}' n'est pas référencé dans inventory.json."
            )
            found_discrepancies.append(disc)
            self.discrepancies.append(disc)
            return found_discrepancies

        doc_id = str(inv["id"])

        # 1. Vérification de la sensibilité
        if "sensitivity_level" in extracted_metadata:
            ext_sens = extracted_metadata["sensitivity_level"]
            inv_sens = inv.get("sensitivity_level")
            if ext_sens != inv_sens:
                disc = MetadataDiscrepancy(
                    document_id=doc_id,
                    source_file=file_path.as_posix(),
                    field_name="sensitivity_level",
                    inventory_value=inv_sens,
                    extracted_value=ext_sens,
                    description=(
                        f"Divergence de sensibilité pour {doc_id} : le fichier déclare {ext_sens} "
                        f"alors qu'inventory.json déclare {inv_sens}."
                    )
                )
                found_discrepancies.append(disc)
                self.discrepancies.append(disc)

        # 2. Vérification de la version / date
        if "date_version" in extracted_metadata:
            ext_ver = str(extracted_metadata["date_version"]).strip()
            inv_ver = str(inv.get("version", "")).strip()
            inv_date = str(inv.get("publication_date", "")).strip()
            if ext_ver != inv_ver and ext_ver != inv_date:
                disc = MetadataDiscrepancy(
                    document_id=doc_id,
                    source_file=file_path.as_posix(),
                    field_name="date_version",
                    inventory_value=f"version={inv_ver}, publication_date={inv_date}",
                    extracted_value=ext_ver,
                    description=(
                        f"Divergence de version pour {doc_id} : le fichier déclare '{ext_ver}' "
                        f"alors qu'inventory.json déclare version='{inv_ver}' et date='{inv_date}'."
                    )
                )
                found_discrepancies.append(disc)
                self.discrepancies.append(disc)

        # 3. Vérification du type documentaire
        if "doc_type" in extracted_metadata:
            ext_type = str(extracted_metadata["doc_type"]).lower()
            inv_type = str(inv.get("type", "")).lower()
            if ext_type != inv_type and not (ext_type == "md" and inv_type == "markdown"):
                disc = MetadataDiscrepancy(
                    document_id=doc_id,
                    source_file=file_path.as_posix(),
                    field_name="doc_type",
                    inventory_value=inv_type,
                    extracted_value=ext_type,
                    description=f"Divergence de type documentaire pour {doc_id} : '{ext_type}' vs '{inv_type}'."
                )
                found_discrepancies.append(disc)
                self.discrepancies.append(disc)

        return found_discrepancies
