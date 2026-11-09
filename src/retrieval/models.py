"""Modèles de données, contextes d'autorisation et exceptions pour le Retriever sémantique."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Any, Optional, Tuple


class ClearanceLevel(IntEnum):
    """Niveaux d'accréditation RBAC d'un utilisateur au sein de DataNova Technologies."""
    PUBLIC = 1      # Accès public / ouvert : documentation générale, catalogues ouverts
    INTERNAL = 2    # Accès interne : architectures, politiques d'entreprise, standards
    RESTRICTED = 3  # Accès restreint / confidentiel : données financières, RH, sécurité SOC


# Correspondance officielle et immuable entre les rôles d'entreprise et les niveaux d'accréditation
ROLE_PERMISSIONS: Dict[str, ClearanceLevel] = {
    # Niveau 1 - Public (Invités, partenaires externes, profils non authentifiés)
    "guest": ClearanceLevel.PUBLIC,
    "anonymous": ClearanceLevel.PUBLIC,
    "external_auditor": ClearanceLevel.PUBLIC,
    "public": ClearanceLevel.PUBLIC,

    # Niveau 2 - Interne (Collaborateurs DataNova, ingénierie, opérations)
    "employee": ClearanceLevel.INTERNAL,
    "data_engineer": ClearanceLevel.INTERNAL,
    "data_scientist": ClearanceLevel.INTERNAL,
    "mlops_engineer": ClearanceLevel.INTERNAL,
    "devops_engineer": ClearanceLevel.INTERNAL,
    "software_engineer": ClearanceLevel.INTERNAL,
    "internal": ClearanceLevel.INTERNAL,

    # Niveau 3 - Restreint / Confidentiel (Sécurité, DSI, Direction, RH, Finance)
    "security_admin": ClearanceLevel.RESTRICTED,
    "ciso": ClearanceLevel.RESTRICTED,
    "dpo": ClearanceLevel.RESTRICTED,
    "executive": ClearanceLevel.RESTRICTED,
    "finance_controller": ClearanceLevel.RESTRICTED,
    "hr_director": ClearanceLevel.RESTRICTED,
    "admin": ClearanceLevel.RESTRICTED,
    "restricted": ClearanceLevel.RESTRICTED
}


class InvalidClearanceError(ValueError):
    """Exception levée lorsqu'un niveau d'accréditation est invalide, mal typé ou hors bornes."""
    pass


class InvalidTopKError(ValueError):
    """Exception levée lorsqu'un paramètre top_k est invalide (type non entier, <= 0 ou hors limites)."""
    pass


class RBACSecurityError(Exception):
    """Exception levée en cas de tentative de violation de sécurité ou de corruption de métadonnées."""
    pass


class RetrieverError(Exception):
    """Exception levée lors d'une défaillance du sous-système de recherche documentaire."""
    pass


# Limite maximale autorisée pour top_k afin de prévenir les dénis de service et l'épuisement mémoire
MAX_TOP_K: int = 50
DEFAULT_TOP_K: int = 4


@dataclass(frozen=True)
class UserContext:
    """
    Contexte d'autorisation immuable et validé représentant l'identité de l'appelant.
    
    Règles de sécurité :
      - Construit et validé EXCLUSIVEMENT côté backend (aucune confiance dans les paramètres client).
      - L'identifiant 'user_id' doit être une chaîne non vide.
      - 'clearance_level' est dérivé côté serveur des rôles certifiés ou fixé par une autorité fiable.
      - Le niveau 'clearance_level' doit appartenir strictement à {1, 2, 3}.
    """
    user_id: str
    clearance_level: int
    roles: Tuple[str, ...] = field(default_factory=tuple)
    department: Optional[str] = None

    def __post_init__(self):
        # 1. Validation de l'identifiant
        if not self.user_id or not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("L'identifiant utilisateur (user_id) ne peut pas être vide.")

        # 2. Validation stricte du type de clearance_level (exclusion des booléens)
        if isinstance(self.clearance_level, bool) or not isinstance(self.clearance_level, int):
            raise InvalidClearanceError(
                f"Le niveau d'accréditation doit être un entier strict. "
                f"Type reçu : '{type(self.clearance_level).__name__}'."
            )

        # 3. Validation de l'intervalle [1, 3]
        if self.clearance_level not in (1, 2, 3):
            raise InvalidClearanceError(
                f"Niveau d'accréditation invalide : {self.clearance_level}. "
                f"Les niveaux autorisés sont strictement 1 (Public), 2 (Interne) ou 3 (Restreint)."
            )

        # 4. Normalisation des rôles en tuple immuable
        if not isinstance(self.roles, tuple):
            object.__setattr__(self, "roles", tuple(self.roles))

    @classmethod
    def from_roles(
        cls,
        user_id: str,
        roles: List[str],
        department: Optional[str] = None
    ) -> "UserContext":
        """
        Construit un UserContext côté backend en dérivant automatiquement le niveau d'accréditation
        maximal à partir de la matrice des rôles certifiés de l'entreprise (Principe du moindre privilège).
        """
        if not roles:
            # Sans rôle explicite : attribution par défaut du niveau le plus bas (Public)
            return cls(
                user_id=user_id,
                clearance_level=ClearanceLevel.PUBLIC.value,
                roles=("guest",),
                department=department
            )

        # Calcul du niveau d'accréditation maximal accordé par les rôles reconnus
        derived_clearance = ClearanceLevel.PUBLIC.value
        normalized_roles: List[str] = []

        for r in roles:
            if not isinstance(r, str):
                continue
            clean_role = r.strip().lower()
            normalized_roles.append(clean_role)
            assigned_level = ROLE_PERMISSIONS.get(clean_role, ClearanceLevel.PUBLIC)
            if assigned_level.value > derived_clearance:
                derived_clearance = assigned_level.value

        return cls(
            user_id=user_id,
            clearance_level=derived_clearance,
            roles=tuple(normalized_roles),
            department=department
        )

    @classmethod
    def public(cls, user_id: str = "guest_user", roles: Optional[List[str]] = None) -> "UserContext":
        """Usine pour profil Niveau 1 (Public)."""
        return cls(user_id=user_id, clearance_level=ClearanceLevel.PUBLIC.value, roles=tuple(roles or ["guest"]))

    @classmethod
    def internal(cls, user_id: str, roles: Optional[List[str]] = None, department: Optional[str] = None) -> "UserContext":
        """Usine pour profil Niveau 2 (Interne)."""
        return cls(user_id=user_id, clearance_level=ClearanceLevel.INTERNAL.value, roles=tuple(roles or ["employee"]), department=department)

    @classmethod
    def restricted(cls, user_id: str, roles: Optional[List[str]] = None, department: Optional[str] = None) -> "UserContext":
        """Usine pour profil Niveau 3 (Restreint / Confidentiel)."""
        return cls(user_id=user_id, clearance_level=ClearanceLevel.RESTRICTED.value, roles=tuple(roles or ["security_admin"]), department=department)

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation sécurisée du contexte d'autorisation."""
        return {
            "user_id": self.user_id,
            "clearance_level": self.clearance_level,
            "roles": list(self.roles),
            "department": self.department
        }


@dataclass(frozen=True)
class RetrievedChunk:
    """
    Représente un chunk documentaire pertinent extrait et formellement validé par le contrôle RBAC.
    
    Attributs géométriques :
      - distance : Distance cosinus brute calculée par ChromaDB (domaine théorique : [0.0, 2.0]).
      - similarity_score : Similarité cosinus calculée par 1 - distance (domaine théorique : [-1.0, 1.0]).
    """
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    distance: float
    sensitivity_level: int

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation sans exposition d'éléments internes."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "similarity_score": self.similarity_score,
            "distance": self.distance,
            "sensitivity_level": self.sensitivity_level
        }


@dataclass(frozen=True)
class RetrievalResult:
    """
    Résultat consolidé et sécurisé d'une opération de recherche documentaire.
    Contient EXCLUSIVEMENT des chunks autorisés pour le contexte de l'appelant.
    """
    query: str
    user_id: str
    clearance_level: int
    chunks: List[RetrievedChunk]
    total_retrieved: int
    top_k_requested: int
    execution_time_ms: float
    filter_applied: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation complète pour les métriques MLOps et l'API."""
        return {
            "query": self.query,
            "user_id": self.user_id,
            "clearance_level": self.clearance_level,
            "total_retrieved": self.total_retrieved,
            "top_k_requested": self.top_k_requested,
            "execution_time_ms": self.execution_time_ms,
            "filter_applied": self.filter_applied,
            "chunks": [c.to_dict() for c in self.chunks]
        }

    def to_context_str(self, separator: str = "\n\n---\n\n") -> str:
        """
        Génère le bloc textuel documentaire prêt à être injecté dans le contexte du LLM (Phase 8).
        Inclut la traçabilité des sources tout en préservant le texte exact.
        """
        if not self.chunks:
            return ""

        blocks: List[str] = []
        for idx, c in enumerate(self.chunks, 1):
            source = c.metadata.get("source", "source_inconnue")
            doc_id = c.metadata.get("document_id", "doc_inconnu")
            blocks.append(
                f"[Document {idx} | ID: {c.chunk_id} | Ref: {doc_id} | Source: {source} | Sensibilité: {c.sensitivity_level}]\n"
                f"{c.content}"
            )
        return separator.join(blocks)
