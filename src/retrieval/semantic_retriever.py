"""Retriever sémantique orchestrant la vectorisation E5, le préfiltrage ChromaDB et le contrôle RBAC."""
import logging
import time
from typing import Optional, Dict, Any, List

from src.embeddings.embedding_config import EmbeddingConfig
from src.embeddings.embedding_manager import EmbeddingManager
from src.embeddings.chroma_store import ChromaStore
from src.retrieval.models import (
    ClearanceLevel,
    UserContext,
    RetrievedChunk,
    RetrievalResult,
    InvalidTopKError,
    RetrieverError,
    MAX_TOP_K,
    DEFAULT_TOP_K
)
from src.retrieval.rbac_manager import RBACManager

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Composant de recherche documentaire sémantique pour le RAG multi-sources de DataNova Technologies.
    
    Garanties d'exécution :
      1. Vectorisation asymétrique : Utilise obligatoirement le préfixe 'query: ' via EmbeddingManager.
      2. Barrière 1 (Pre-filtering ChromaDB) : Applique 'where={"sensitivity_level": {"$lte": clearance}}'.
      3. Barrière 2 (Défense en profondeur) : Valide applicativement chaque chunk extrait (Zero Trust).
      4. Politique de Default-Deny : Exclusion immédiate de tout chunk aux métadonnées douteuses ou absentes.
      5. Calcul rigoureux du score : similarity_score = 1 - distance (espace cosinus, domaine [-1, 1]).
      6. Validation stricte des entrées : Requêtes non vides, top_k borné dans [1, MAX_TOP_K].
      7. Traçabilité complète : Temps d'exécution, filtres appliqués et métadonnées de provenance préservées.
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        chroma_store: Optional[ChromaStore] = None,
        rbac_manager: Optional[RBACManager] = None,
        default_top_k: int = DEFAULT_TOP_K
    ):
        self.config = EmbeddingConfig.create_structural_config()
        self.embedding_manager = embedding_manager or EmbeddingManager(config=self.config)
        self.chroma_store = chroma_store or ChromaStore(config=self.config)
        self.rbac_manager = rbac_manager or RBACManager()
        self.default_top_k = self._validate_top_k(default_top_k)

    @staticmethod
    def _validate_top_k(top_k: Any) -> int:
        """Valide rigoureusement le paramètre top_k."""
        if isinstance(top_k, bool):
            raise InvalidTopKError("Le paramètre top_k ne peut pas être un booléen.")
        if not isinstance(top_k, int):
            raise InvalidTopKError(
                f"Le paramètre top_k doit être un entier strict (reçu '{type(top_k).__name__}')."
            )
        if top_k <= 0:
            raise InvalidTopKError(
                f"Le paramètre top_k doit être strictement supérieur à 0 (reçu {top_k})."
            )
        if top_k > MAX_TOP_K:
            raise InvalidTopKError(
                f"Le paramètre top_k ({top_k}) dépasse la limite maximale autorisée ({MAX_TOP_K})."
            )
        return top_k

    def retrieve(
        self,
        query: str,
        user_context: UserContext,
        top_k: Optional[int] = None,
        min_similarity_score: Optional[float] = None
    ) -> RetrievalResult:
        """
        Exécute une recherche sémantique sécurisée pour un utilisateur donné.
        
        Étapes du pipeline :
          1. Validation de la requête et de l'identité utilisateur.
          2. Contrôle de vacuité de la base vectorielle.
          3. Calcul de l'embedding de requête (préfixe 'query: ').
          4. Interrogation ChromaDB avec préfiltrage 'where'.
          5. Post-validation RBAC individuelle de chaque chunk (Default-Deny).
          6. Conversion des distances cosinus en scores de similarité (1 - distance).
          7. Construction du RetrievalResult exploitable par la Phase 8.
        """
        start_time = time.perf_counter()

        # 1. Validation de la requête
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("La requête de recherche ne peut pas être vide ou composée uniquement d'espaces.")
        clean_query = query.strip()

        # 2. Validation du contexte utilisateur
        if not isinstance(user_context, UserContext):
            raise TypeError(
                f"Le contexte utilisateur doit être une instance de UserContext (reçu '{type(user_context).__name__}')."
            )

        # 3. Validation de top_k
        effective_top_k = self._validate_top_k(top_k) if top_k is not None else self.default_top_k

        # 4. Construction du filtre RBAC (Barrière 1 : Pre-filtering)
        where_filter = self.rbac_manager.build_chroma_filter(user_context.clearance_level)

        # 5. Contrôle de vacuité de la base ChromaDB
        try:
            total_docs = self.chroma_store.count()
        except Exception as e:
            raise RetrieverError(f"Impossible de vérifier l'état du store ChromaDB : {e}")

        if total_docs == 0:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.info("Retriever : La collection vectorielle est vide (0 document).")
            return RetrievalResult(
                query=clean_query,
                user_id=user_context.user_id,
                clearance_level=user_context.clearance_level,
                chunks=[],
                total_retrieved=0,
                top_k_requested=effective_top_k,
                execution_time_ms=elapsed_ms,
                filter_applied=where_filter
            )

        # Clamping défensif du nombre de résultats demandés à ChromaDB
        n_results = min(effective_top_k, total_docs)

        # 6. Vectorisation de la requête avec préfixe 'query: '
        try:
            query_vector = self.embedding_manager.embed_query(clean_query)
        except Exception as e:
            raise RetrieverError(f"Échec de vectorisation de la requête utilisateur : {e}")

        # 7. Interrogation ChromaDB avec préfiltrage natif
        try:
            raw_results = self.chroma_store.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            raise RetrieverError(f"Échec lors de l'exécution de la requête ChromaDB : {e}")

        # 8. Traitement des résultats bruts et Barrière 2 (Défense en profondeur)
        authorized_chunks: List[RetrievedChunk] = []

        if raw_results and raw_results.get("ids") and raw_results["ids"][0]:
            ids = raw_results["ids"][0]
            docs = raw_results["documents"][0] if raw_results.get("documents") else []
            metas = raw_results["metadatas"][0] if raw_results.get("metadatas") else []
            dists = raw_results["distances"][0] if raw_results.get("distances") else []

            for chunk_id, content, meta, dist in zip(ids, docs, metas, dists):
                # Barrière 2 : Validation défensive de sécurité
                if not self.rbac_manager.is_access_granted(meta, user_context.clearance_level):
                    # Journalisation sécurisée sans fuite de contenu
                    continue

                # Calcul rigoureux du score de similarité cosinus (domaine théorique [-1.0, 1.0])
                float_dist = float(dist)
                similarity = round(1.0 - float_dist, 4)

                # Filtrage optionnel par seuil de score minimal
                if min_similarity_score is not None and similarity < min_similarity_score:
                    continue

                authorized_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        content=content,
                        metadata=dict(meta) if meta else {},
                        similarity_score=similarity,
                        distance=round(float_dist, 4),
                        sensitivity_level=int(meta["sensitivity_level"])
                    )
                )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.info(
            f"Retriever : {len(authorized_chunks)} chunks autorisés extraits en {elapsed_ms} ms "
            f"[user_id='{user_context.user_id}', clearance={user_context.clearance_level}, top_k={effective_top_k}]."
        )

        return RetrievalResult(
            query=clean_query,
            user_id=user_context.user_id,
            clearance_level=user_context.clearance_level,
            chunks=authorized_chunks,
            total_retrieved=len(authorized_chunks),
            top_k_requested=effective_top_k,
            execution_time_ms=elapsed_ms,
            filter_applied=where_filter
        )


# ==============================================================================
# SCRIPT DE DÉMONSTRATION ET D'AUDIT CLI (POWERSHELL)
# ==============================================================================

if __name__ == "__main__":
    import json

    print("\n" + "=" * 80)
    print("DATANOVA TECHNOLOGIES — RETRIEVER SÉMANTIQUE & CONTRÔLE D'ACCÈS RBAC (PHASE 7)")
    print("=" * 80)

    # Initialisation du retriever connecté à la base ChromaDB existante
    retriever = SemanticRetriever()
    total_stored = retriever.chroma_store.count()
    print(f"\n[*] Connexion à ChromaDB : {total_stored} chunks indexés dans '{retriever.config.collection_name}'.")

    # Scénarios de tests avec trois profils utilisateurs
    scenarios = [
        (
            "Question 1 (Profil Public / Invité)",
            "Quels sont les datasets disponibles pour la télémétrie IoT ?",
            UserContext.from_roles("user_guest", ["guest"])
        ),
        (
            "Question 2 (Profil Interne / Data Engineer)",
            "Quelles sont les procédures de gouvernance et les standards d'ingénierie ?",
            UserContext.from_roles("user_engineer", ["data_engineer"])
        ),
        (
            "Question 3 (Profil Restreint / CISO & Direction)",
            "Quels sont les journaux d'audit de sécurité et données de facturation sensibles ?",
            UserContext.from_roles("user_ciso", ["security_admin", "ciso"])
        )
    ]

    for title, query_text, user_ctx in scenarios:
        print("\n" + "-" * 80)
        print(f"Scénario : {title}")
        print(f"Utilisateur  : {user_ctx.user_id} (Rôles: {user_ctx.roles})")
        print(f"Accréditation : Niveau {user_ctx.clearance_level} ({ClearanceLevel(user_ctx.clearance_level).name})")
        print(f"Question     : \"{query_text}\"")

        result = retriever.retrieve(query=query_text, user_context=user_ctx, top_k=3)
        print(f"Résultats    : {result.total_retrieved} chunks extraits en {result.execution_time_ms} ms")
        print(f"Filtre natif : {json.dumps(result.filter_applied)}")

        for idx, chunk in enumerate(result.chunks, 1):
            doc_id = chunk.metadata.get("document_id", "N/A")
            sens = chunk.sensitivity_level
            print(f"  [{idx}] Chunk: {chunk.chunk_id} | Doc: {doc_id} | Sensibilité: {sens} | Sim: {chunk.similarity_score:.4f} | Dist: {chunk.distance:.4f}")
            # Aperçu tronqué pour affichage propre
            apercu = chunk.content.replace("\n", " ")[:90]
            print(f"      Extrait : {apercu}...")

    print("\n" + "=" * 80)
    print("FIN DE LA DÉMONSTRATION DU RETRIEVER SÉMANTIQUE RBAC")
    print("=" * 80 + "\n")

