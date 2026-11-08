"""Validateur strict et qualitatif des documents normalisés pour le pipeline d'ingestion."""
import logging
from typing import List, Dict, Any, Tuple, Optional
from src.ingestion.models import NormalizedDocument

logger = logging.getLogger(__name__)


class DocumentValidator:
    """
    Validateur documentaire pour le corpus DataNova Technologies.
    Garantit l'intégrité absolue de chaque NormalizedDocument sans altérer ses métadonnées :
      1. Contenu textuel non vide et exploitable.
      2. Métadonnées universelles obligatoires :
         'document_id', 'source', 'date_version', 'doc_type', 'sensitivity_level'.
      3. Métadonnées spécifiques obligatoires par format :
         - PDF : 'page_number'
         - FAQ : 'faq_id' et 'category'
         - CSV : 'dataset_id' et 'domain'
         - Markdown : provenance et métadonnées obligatoires
      4. Distinction stricte entre métadonnées obligatoires et facultatives.
      5. Sensibilité strictement entière dans [1, 2, 3].
      6. Compatibilité ChromaDB : valeurs primitives uniquement (str, int, float, bool).
      7. Conservation du contexte d'erreur précis (fichier, id, page).
    """

    UNIVERSAL_MANDATORY_FIELDS = [
        "document_id",
        "source",
        "date_version",
        "doc_type",
        "sensitivity_level"
    ]

    ALLOWED_DOC_TYPES = {"pdf", "markdown", "faq", "csv"}

    @classmethod
    def validate(cls, doc: NormalizedDocument) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Valide un document unitaire de manière non destructive.
        Retourne (is_valid, error_reason, warning_reason).
        """
        if not isinstance(doc, NormalizedDocument):
            return False, f"L'objet fourni n'est pas une instance de NormalizedDocument (type: {type(doc).__name__}).", None

        # 1. Validation de l'identifiant unitaire
        if not doc.id or not str(doc.id).strip():
            return False, "L'identifiant 'id' du document est vide ou manquant.", None

        # 2. Validation du contenu textuel
        if not doc.page_content or not doc.page_content.strip():
            return False, f"Le document '{doc.id}' a un contenu textuel (page_content) vide.", None

        meta = doc.metadata or {}

        # 3. Validation des 5 métadonnées universelles obligatoires
        missing_universal = [
            f for f in cls.UNIVERSAL_MANDATORY_FIELDS
            if f not in meta or meta[f] is None or str(meta[f]).strip() == ""
        ]
        if missing_universal:
            return False, f"Document '{doc.id}' (source: {meta.get('source', 'N/A')}) : métadonnées obligatoires manquantes {missing_universal}.", None

        # 4. Validation du type documentaire
        doc_type = str(meta["doc_type"]).lower()
        if doc_type not in cls.ALLOWED_DOC_TYPES:
            return False, f"Document '{doc.id}' : doc_type '{doc_type}' non autorisé (autorisés: {cls.ALLOWED_DOC_TYPES}).", None

        # 5. Validation stricte du niveau de sensibilité
        raw_sens = meta.get("sensitivity_level")
        try:
            sens = int(raw_sens)
            if sens not in (1, 2, 3):
                return False, f"Document '{doc.id}' : sensitivity_level={sens} hors bornes (attendu: 1, 2 ou 3).", None
        except (ValueError, TypeError):
            return False, f"Document '{doc.id}' : sensitivity_level='{raw_sens}' non entier.", None

        # 6. Validation des métadonnées spécifiques obligatoires par format
        warning: Optional[str] = None
        if doc_type == "pdf":
            page_num = meta.get("page_number")
            if page_num is None:
                return False, f"Document PDF '{doc.id}' : métadonnée obligatoire 'page_number' manquante.", None
            try:
                if int(page_num) < 1:
                    return False, f"Document PDF '{doc.id}' : page_number={page_num} invalide (< 1).", None
            except (ValueError, TypeError):
                return False, f"Document PDF '{doc.id}' : page_number='{page_num}' non entier.", None

        elif doc_type == "faq":
            faq_id = meta.get("faq_id")
            if not faq_id or not str(faq_id).strip():
                return False, f"Document FAQ '{doc.id}' : métadonnée obligatoire 'faq_id' manquante ou vide.", None
            category = meta.get("category")
            if not category or not str(category).strip():
                return False, f"Document FAQ '{doc.id}' : métadonnée obligatoire 'category' manquante ou vide.", None

        elif doc_type == "csv":
            dataset_id = meta.get("dataset_id")
            if not dataset_id or not str(dataset_id).strip():
                return False, f"Document CSV '{doc.id}' : métadonnée obligatoire 'dataset_id' manquante ou vide.", None
            domain = meta.get("domain")
            if not domain or not str(domain).strip():
                return False, f"Document CSV '{doc.id}' : métadonnée obligatoire 'domain' manquante ou vide.", None

        elif doc_type == "markdown":
            # Métadonnée facultative : title
            if "title" not in meta or not str(meta["title"]).strip():
                warning = f"Document Markdown '{doc.id}' : métadonnée facultative 'title' non renseignée."

        # 7. Contrôle de compatibilité ChromaDB (toutes les métadonnées doivent être primitives)
        for key, val in meta.items():
            if not isinstance(val, (str, int, float, bool)):
                return False, (
                    f"Document '{doc.id}' : métadonnée '{key}' incompatible avec ChromaDB "
                    f"(type complexe '{type(val).__name__}' détecté, seules les primitives sont admises)."
                ), None

        return True, None, warning

    @classmethod
    def validate_batch(
        cls,
        documents: List[NormalizedDocument]
    ) -> Tuple[List[NormalizedDocument], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Valide un lot de documents et conserve le contexte précis pour les rapports.
        Retourne (documents_valides, details_rejets, details_avertissements).
        """
        valid_docs: List[NormalizedDocument] = []
        rejected_details: List[Dict[str, Any]] = []
        warnings_details: List[Dict[str, Any]] = []

        for doc in documents:
            is_valid, error, warn = cls.validate(doc)
            doc_id = getattr(doc, "id", "INCONNU")
            doc_meta = getattr(doc, "metadata", {})
            source = doc_meta.get("source", "INCONNUE")
            page_num = doc_meta.get("page_number")

            if is_valid:
                valid_docs.append(doc)
                if warn:
                    warnings_details.append({
                        "id": doc_id,
                        "source": source,
                        "page_number": page_num,
                        "warning": warn
                    })
            else:
                logger.warning(f"Validation échec sur {doc_id} ({source}) : {error}")
                rejected_details.append({
                    "id": doc_id,
                    "document_id": doc_meta.get("document_id", "INCONNU"),
                    "source": source,
                    "page_number": page_num,
                    "reason": error
                })

        return valid_docs, rejected_details, warnings_details
