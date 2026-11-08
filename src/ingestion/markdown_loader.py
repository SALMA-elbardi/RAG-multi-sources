"""Loader spécialisé pour l'extraction des documents Markdown avec frontmatter."""
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.ingestion.models import NormalizedDocument
from src.ingestion.metadata_manager import MetadataManager

logger = logging.getLogger(__name__)


class MarkdownLoader:
    """
    Chargeur de documents Markdown pour le corpus DataNova Technologies.
    Règles strictes :
      - Charge les fichiers .md et extrait le contenu textuel structuré.
      - Parse le frontmatter YAML initial sans écrasement silencieux.
      - Signale toute divergence avec inventory.json via MetadataManager.
      - Préserve la structure du document (titres, tableaux, blocs de code).
    """

    def __init__(self, metadata_manager: MetadataManager):
        self.metadata_manager = metadata_manager

    def _parse_frontmatter(self, text: str) -> tuple[Dict[str, Any], str]:
        """Extrait le dictionnaire de frontmatter YAML et le corps textuel restant."""
        frontmatter: Dict[str, Any] = {}
        body = text

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
        if match:
            raw_yaml = match.group(1)
            body = match.group(2)
            for line in raw_yaml.splitlines():
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    # Conversion des entiers
                    if v.isdigit():
                        frontmatter[k] = int(v)
                    else:
                        frontmatter[k] = v

        return frontmatter, body

    def load(self, file_path: Path) -> List[NormalizedDocument]:
        """Extrait et normalise le document Markdown."""
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier Markdown introuvable : {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Impossible de lire le fichier Markdown '{file_path}' : {e}")

        clean_raw = content.strip()
        if not clean_raw:
            raise ValueError(f"Le fichier Markdown '{file_path}' est vide.")

        frontmatter, body_text = self._parse_frontmatter(clean_raw)
        clean_body = body_text.strip()
        if not clean_body:
            raise ValueError(f"Le corps textuel du fichier Markdown '{file_path}' est vide après parsing.")

        # Réconciliation avec inventory.json et détection des divergences
        inv = self.metadata_manager.get_by_path(file_path)
        self.metadata_manager.detect_discrepancies(file_path, frontmatter)

        if not inv:
            logger.warning(f"Le fichier {file_path} n'est pas répertorié dans inventory.json.")
            doc_id = str(frontmatter.get("id") or file_path.stem)
            sensitivity = int(frontmatter.get("sensitivity_level", 2))
            version = str(frontmatter.get("date_version", "1.0.0"))
            title = file_path.stem
            owner_team = ""
        else:
            doc_id = str(inv["id"])
            sensitivity = int(inv["sensitivity_level"])
            version = str(inv.get("version", frontmatter.get("date_version", "1.0.0")))
            title = str(inv.get("title", file_path.stem))
            owner_team = str(inv.get("owner_team", ""))

        source_path = file_path.as_posix()
        if "data/raw/" in source_path:
            source_path = "data/raw/" + source_path.split("data/raw/")[-1]

        metadata: Dict[str, Any] = {
            "document_id": doc_id,
            "source": source_path,
            "date_version": version,
            "doc_type": "markdown",
            "sensitivity_level": sensitivity,
            "title": title,
            "owner_team": owner_team
        }

        # Pour le Markdown, l'identifiant unitaire 'id' correspond au document_id officiel
        doc = NormalizedDocument(
            id=doc_id,
            page_content=clean_body,
            metadata=metadata
        )
        return [doc]

