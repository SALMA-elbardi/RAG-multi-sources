"""Validateur de qualité, d'intégrité, de sécurité et d'unicité des chunks."""
import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict

from src.chunking.models import ChunkDocument, ChunkValidationReport

logger = logging.getLogger(__name__)


class ChunkValidator:
    """
    Validateur rigoureux des chunks documentaires générés pour DataNova Technologies.
    
    Contrôles stricts de validité (rejet si non respecté) :
      - Contenu textuel non vide et non purement décoratif (pas de '---' ou '***' isolés).
      - Présence des 5 métadonnées obligatoires universelles (document_id, source, date_version, doc_type, sensitivity_level).
      - Présence des métadonnées obligatoires de chunking (chunk_id, chunk_index, total_chunks, chunking_strategy).
      - Niveau de sensibilité RBAC strictement compris dans {1, 2, 3}.
      - Types de métadonnées exclusivement primitifs (compatibilité ChromaDB).
      - Unicité absolue des identifiants (chunk_id) dans le lot traité (zéro collision).
      - Cohérence de l'index séquentiel (0 <= chunk_index < total_chunks).
      
    Avertissements qualitatifs (sans rejet) :
      - Chunks très courts (< min_chunk_size).
      - Chunks dépassant la taille indicative (soft_max_size).
      - Détection de doublons textuels exacts au sein du même document parent.
    """

    MANDATORY_METADATA_KEYS = {
        "document_id",
        "source",
        "date_version",
        "doc_type",
        "sensitivity_level",
        "chunk_id",
        "chunk_index",
        "total_chunks",
        "chunking_strategy"
    }

    ALLOWED_DOC_TYPES = {"pdf", "markdown", "faq", "csv"}

    def __init__(self, min_chunk_size: int = 40, soft_max_size: int = 700):
        self.min_chunk_size = min_chunk_size
        self.soft_max_size = soft_max_size

    def validate_chunk(self, chunk: ChunkDocument) -> Tuple[bool, Optional[str], List[str]]:
        """
        Valide un chunk individuel.
        Retourne (is_valid, reject_reason, warnings).
        """
        warnings: List[str] = []

        # 1. Contenu non vide et non purement décoratif
        clean_content = chunk.content.strip() if chunk.content else ""
        if not clean_content:
            return False, "Contenu textuel du chunk vide ou constitué d'espaces.", warnings

        # Rejet des fragments purement décoratifs (séparateurs horizontaux isolés '---', '***', '___')
        if re.fullmatch(r"[-*_\s=]{2,}", clean_content):
            return False, "Le chunk est constitué uniquement de séparateurs horizontaux décoratifs sans valeur sémantique.", warnings

        # 2. Métadonnées obligatoires
        meta = chunk.metadata
        for key in self.MANDATORY_METADATA_KEYS:
            if key not in meta or meta[key] is None or str(meta[key]).strip() == "":
                return False, f"Métadonnée obligatoire '{key}' manquante ou vide.", warnings

        # 3. Contrôle doc_type
        doc_type = str(meta.get("doc_type", "")).lower()
        if doc_type not in self.ALLOWED_DOC_TYPES:
            return False, f"doc_type invalide '{doc_type}' (attendu: {self.ALLOWED_DOC_TYPES}).", warnings

        # 4. Contrôle sensibilité RBAC
        try:
            sens = int(meta["sensitivity_level"])
            if sens not in (1, 2, 3):
                return False, f"Niveau de sensibilité invalide {sens} (attendu: 1, 2 ou 3).", warnings
        except (ValueError, TypeError):
            return False, f"Valeur de sensibilité non entière : {meta.get('sensitivity_level')}", warnings

        # 5. Cohérence chunk_index et total_chunks
        try:
            idx = int(meta["chunk_index"])
            total = int(meta["total_chunks"])
            if total < 1:
                return False, f"total_chunks invalide ({total} < 1).", warnings
            if idx < 0 or idx >= total:
                return False, f"chunk_index ({idx}) hors de l'intervalle [0, {total - 1}].", warnings
        except (ValueError, TypeError):
            return False, "Index séquentiel ou total_chunks non entier.", warnings

        # 6. Types primitifs stricts pour ChromaDB
        for k, v in meta.items():
            if not isinstance(v, (str, int, float, bool)):
                return False, f"Métadonnée '{k}' de type non primitif '{type(v).__name__}' incompatible avec ChromaDB.", warnings

        # 7. Contrôle des métadonnées spécifiques (avertissements)
        if doc_type == "pdf" and "page_number" not in meta:
            warnings.append(f"Chunk PDF {chunk.chunk_id} sans 'page_number'.")
        elif doc_type == "faq" and "faq_id" not in meta:
            warnings.append(f"Chunk FAQ {chunk.chunk_id} sans 'faq_id'.")
        elif doc_type == "csv" and "dataset_id" not in meta:
            warnings.append(f"Chunk CSV {chunk.chunk_id} sans 'dataset_id'.")

        # 8. Alertes de taille (informatif, non bloquant)
        content_len = len(clean_content)
        if content_len < self.min_chunk_size:
            warnings.append(f"Chunk très court ({content_len} car. < {self.min_chunk_size}) : '{chunk.chunk_id}'.")
        elif content_len > self.soft_max_size:
            warnings.append(f"Chunk dépassant la taille indicative ({content_len} car. > {self.soft_max_size}) : '{chunk.chunk_id}'.")

        return True, None, warnings

    def validate_batch(self, chunks: List[ChunkDocument]) -> Tuple[List[ChunkDocument], ChunkValidationReport]:
        """
        Valide un ensemble complet de chunks avec contrôles globaux d'unicité et de doublons.
        Retourne (chunks_valides, rapport).
        """
        report = ChunkValidationReport()
        report.total_chunks_inspected = len(chunks)

        valid_chunks: List[ChunkDocument] = []
        seen_chunk_ids: Set[str] = set()
        doc_texts: Dict[str, Set[str]] = defaultdict(set)
        chunks_per_parent: Dict[str, int] = defaultdict(int)

        for chunk in chunks:
            is_valid, reject_reason, chunk_warnings = self.validate_chunk(chunk)

            # Vérification de l'unicité globale des chunk_id (rejet en cas de collision)
            if chunk.chunk_id in seen_chunk_ids:
                is_valid = False
                reject_reason = f"Collision d'identifiant détectée : '{chunk.chunk_id}' apparaît plusieurs fois."
            else:
                seen_chunk_ids.add(chunk.chunk_id)

            if not is_valid:
                report.rejected_chunks_count += 1
                report.rejected_details.append({
                    "chunk_id": chunk.chunk_id,
                    "reason": reject_reason,
                    "preview": chunk.content[:60] if chunk.content else ""
                })
                continue

            # Détection de doublons textuels au sein du même document parent
            parent_id = chunk.document_id
            content_norm = " ".join(chunk.content.strip().split())
            if content_norm in doc_texts[parent_id]:
                chunk_warnings.append(f"Contenu textuel dupliqué détecté dans le document parent '{parent_id}'.")
            else:
                doc_texts[parent_id].add(content_norm)

            chunks_per_parent[parent_id] += 1

            if chunk_warnings:
                report.warnings_count += len(chunk_warnings)
                for w in chunk_warnings:
                    report.warnings.append({
                        "chunk_id": chunk.chunk_id,
                        "warning": w
                    })

            valid_chunks.append(chunk)

        report.valid_chunks_count = len(valid_chunks)
        return valid_chunks, report
