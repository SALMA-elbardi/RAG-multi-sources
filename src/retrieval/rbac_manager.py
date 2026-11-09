"""Gestionnaire de contrôle d'accès RBAC (Role-Based Access Control) et politique de sécurité documentaire."""
import logging
from typing import Dict, Any, Optional, Tuple, List

from src.retrieval.models import (
    ClearanceLevel,
    InvalidClearanceError,
    ROLE_PERMISSIONS,
    RetrievedChunk
)

logger = logging.getLogger(__name__)


class RBACManager:
    """
    Gestionnaire centralisant la politique de sécurité documentaire de DataNova Technologies.
    
    Principes directeurs :
      1. Règle d'accès : sensitivity_level <= clearance_level
         - Niveau 1 (Public) : Accès exclusif aux documents de niveau 1.
         - Niveau 2 (Interne) : Accès aux documents de niveaux 1 et 2.
         - Niveau 3 (Restreint / Confidentiel) : Accès à l'ensemble des niveaux (1, 2 et 3).
      2. Dérivation serveur : Le niveau d'accréditation est calculé côté backend d'après les rôles certifiés.
      3. Préfiltrage natif ChromaDB : Clause 'where={"sensitivity_level": {"$lte": clearance}}'.
      4. Défense en profondeur : Validation applicative systématique de chaque chunk extrait.
      5. Politique de Default-Deny : Tout chunk sans métadonnée valide est immédiatement écarté.
      6. Journalisation sécurisée : Traces d'audit sans fuite de contenu textuel ou d'informations sensibles.
    """

    VALID_SENSITIVITY_LEVELS = {ClearanceLevel.PUBLIC.value, ClearanceLevel.INTERNAL.value, ClearanceLevel.RESTRICTED.value}
    VALID_CLEARANCE_LEVELS = {ClearanceLevel.PUBLIC.value, ClearanceLevel.INTERNAL.value, ClearanceLevel.RESTRICTED.value}

    def validate_clearance(self, clearance_level: Any) -> int:
        """
        Valide rigoureusement le niveau d'accréditation utilisateur.
        Lève InvalidClearanceError en cas de type non entier, booléen ou valeur hors de {1, 2, 3}.
        """
        if isinstance(clearance_level, bool) or not isinstance(clearance_level, int):
            raise InvalidClearanceError(
                f"Le niveau d'accréditation doit être un entier strict (reçu '{type(clearance_level).__name__}')."
            )

        if clearance_level not in self.VALID_CLEARANCE_LEVELS:
            raise InvalidClearanceError(
                f"Niveau d'accréditation invalide : {clearance_level}. "
                f"Valeurs autorisées : 1 (Public), 2 (Interne) ou 3 (Restreint)."
            )

        return clearance_level

    def resolve_clearance_from_roles(self, roles: List[str]) -> int:
        """
        Dérive le niveau d'accréditation côté serveur à partir d'une liste de rôles certifiés.
        Applique le principe de moindre privilège : retourne Niveau 1 par défaut.
        """
        if not roles:
            return ClearanceLevel.PUBLIC.value

        max_clearance = ClearanceLevel.PUBLIC.value
        for role in roles:
            if not isinstance(role, str):
                continue
            assigned = ROLE_PERMISSIONS.get(role.strip().lower(), ClearanceLevel.PUBLIC)
            if assigned.value > max_clearance:
                max_clearance = assigned.value

        return max_clearance

    def build_chroma_filter(self, clearance_level: int) -> Dict[str, Any]:
        """
        Construit le filtre de métadonnées natif pour l'interrogation ChromaDB (Barrière 1 : Pre-filtering).
        Permet à ChromaDB de restreindre les résultats retournés aux seuls documents respectant l'inégalité.
        """
        valid_clearance = self.validate_clearance(clearance_level)
        return {"sensitivity_level": {"$lte": valid_clearance}}

    def is_access_granted(self, metadata: Optional[Dict[str, Any]], clearance_level: int) -> bool:
        """
        Contrôle défensif de sécurité côté application (Barrière 2 : Zero Trust).
        
        Applique rigoureusement la politique de Default-Deny :
          - Retourne False si les métadonnées sont absentes, None ou corrompues.
          - Retourne False si 'sensitivity_level' est manquant, booléen, non entier ou hors [1, 3].
          - Retourne False si 'sensitivity_level > clearance_level'.
          - Retourne True UNIQUEMENT si 'sensitivity_level <= clearance_level'.
        
        Journalisation sécurisée :
          - Enregistre uniquement les métadonnées techniques (chunk_id, doc_id, doc_type, niveaux).
          - Ne journalise JAMAIS le contenu textuel du document.
        """
        valid_clearance = self.validate_clearance(clearance_level)

        # 1. Contrôle d'existence du conteneur de métadonnées
        if not metadata or not isinstance(metadata, dict):
            logger.warning("Sécurité RBAC (Default-Deny) : Métadonnées absentes ou non dictionnaires. Accès refusé.")
            return False

        chunk_id = metadata.get("chunk_id", "sans_id")
        doc_id = metadata.get("document_id", "sans_doc")
        doc_type = metadata.get("doc_type", "inconnu")

        # 2. Contrôle de présence du champ sensitivity_level
        if "sensitivity_level" not in metadata:
            logger.warning(
                f"Sécurité RBAC (Default-Deny) : Champ 'sensitivity_level' absent "
                f"[chunk_id='{chunk_id}', doc_id='{doc_id}', type='{doc_type}']. Accès refusé."
            )
            return False

        raw_level = metadata["sensitivity_level"]

        # 3. Contrôle de type strict (exclusion formelle des booléens)
        if isinstance(raw_level, bool) or not isinstance(raw_level, int):
            logger.warning(
                f"Sécurité RBAC (Default-Deny) : 'sensitivity_level' invalide (type='{type(raw_level).__name__}') "
                f"[chunk_id='{chunk_id}', doc_id='{doc_id}']. Accès refusé."
            )
            return False

        # 4. Contrôle du domaine de valeurs [1, 3]
        if raw_level not in self.VALID_SENSITIVITY_LEVELS:
            logger.warning(
                f"Sécurité RBAC (Default-Deny) : 'sensitivity_level' hors bornes ({raw_level}) "
                f"[chunk_id='{chunk_id}', doc_id='{doc_id}']. Accès refusé."
            )
            return False

        # 5. Règle formelle d'autorisation : sensitivity_level <= clearance_level
        if raw_level > valid_clearance:
            logger.warning(
                f"Sécurité RBAC : Violation interceptée [chunk_id='{chunk_id}', doc_id='{doc_id}', "
                f"sensibilité_requise={raw_level}, accréditation_utilisateur={valid_clearance}]. Accès refusé."
            )
            return False

        return True

    def filter_chunks_defensive(
        self,
        chunks: List[RetrievedChunk],
        clearance_level: int
    ) -> Tuple[List[RetrievedChunk], int]:
        """
        Passe en revue une liste de RetrievedChunk et élimine défensivement tout chunk non autorisé.
        Retourne : (chunks_autorisés, nombre_de_chunks_écartés).
        """
        valid_clearance = self.validate_clearance(clearance_level)
        authorized: List[RetrievedChunk] = []
        rejected_count: int = 0

        for chunk in chunks:
            if self.is_access_granted(chunk.metadata, valid_clearance):
                authorized.append(chunk)
            else:
                rejected_count += 1

        return authorized, rejected_count
