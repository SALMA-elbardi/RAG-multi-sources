"""Gestionnaire de persistance vectorielle ChromaDB avec isolation et reconstruction sécurisée."""
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import chromadb
from chromadb.api.models.Collection import Collection

from src.embeddings.embedding_config import EmbeddingConfig

logger = logging.getLogger(__name__)

# Types primitifs strictement autorisés par ChromaDB
PrimitiveMetadataValue = Union[str, int, float, bool]


class ChromaStore:
    """
    Gestionnaire du stockage vectoriel ChromaDB pour le RAG multi-sources de DataNova Technologies.
    
    Garanties :
      1. Instance persistante unique hébergeant deux collections isolées (structural / fixed).
      2. Nettoyage strict des métadonnées (omission des valeurs None sans invention de valeurs).
      3. Indexation incrémentale idempotente via `upsert`.
      4. Reconstruction complète sécurisée par collection intermédiaire (*Blue/Green Staging*).
      5. Préservation absolue de la collection concurrente lors des opérations de maintenance.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig.create_structural_config()
        self.persist_directory = Path(self.config.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialisation du client persistant ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection: Optional[Collection] = None

    @property
    def collection(self) -> Collection:
        """Récupération ou création paresseuse de la collection vectorielle cible."""
        if self._collection is None:
            metadata = {
                "hnsw:space": self.config.distance_metric,
                "model_name": self.config.model_name,
                "model_revision": self.config.model_revision,
                "dimension": self.config.embedding_dim,
                "strategy": self.config.chunking_strategy
            }
            self._collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata=metadata
            )
        return self._collection

    @staticmethod
    def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, PrimitiveMetadataValue]:
        """
        Nettoie défensivement le dictionnaire de métadonnées pour compatibilité stricte ChromaDB :
          - Omet toute clé dont la valeur est None (pas d'invention de valeur artificielle).
          - Conserve les types primitifs (str, int, float, bool).
          - Lève une exception explicite si un type complexe (list, dict) est rencontré.
        """
        clean: Dict[str, PrimitiveMetadataValue] = {}
        for key, val in metadata.items():
            if val is None:
                # Omission stricte : aucune valeur inventée
                continue

            if isinstance(val, (str, int, float, bool)):
                clean[key] = val
            else:
                raise ValueError(
                    f"Métadonnée '{key}' de type non primitif '{type(val).__name__}' "
                    f"interdite par ChromaDB."
                )
        return clean

    def upsert_chunks(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> int:
        """
        Indexe ou met à jour des chunks de manière idempotente dans la collection cible.
        Garantit que le préfixe d'inférence n'est PAS stocké dans 'documents'.
        """
        if not ids:
            return 0

        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError(
                f"Incohérence de dimensions dans le lot d'insertion : "
                f"ids={len(ids)}, docs={len(documents)}, embs={len(embeddings)}, metas={len(metadatas)}"
            )

        cleaned_metadatas = [self.clean_metadata(m) for m in metadatas]

        # Insertion idempotente via upsert
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned_metadatas
        )

        logger.info(f"Upsert de {len(ids)} chunks dans la collection '{self.config.collection_name}'.")
        return len(ids)

    def rebuild_collection(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> Tuple[int, str]:
        """
        Reconstruction complète sécurisée de la collection cible via collection Staging (Blue/Green).
        
        Protocole de sécurité :
          1. Crée une collection temporaire `__staging`.
          2. Insère et valide l'ensemble des données dans `__staging`.
          3. Supprime l'ancienne collection cible et la recrée.
          4. Copie les éléments validés vers la collection cible.
          5. Supprime la collection `__staging`.
          6. Ne touche JAMAIS à l'autre collection (ex: datanova_corpus_fixed).
        """
        target_name = self.config.collection_name
        staging_name = f"{target_name}__staging"

        # Étape 1 : Nettoyage d'un éventuel staging orphelin résiduel
        existing_collections = [c.name for c in self.client.list_collections()]
        if staging_name in existing_collections:
            self.client.delete_collection(staging_name)

        staging_metadata = {
            "hnsw:space": self.config.distance_metric,
            "model_name": self.config.model_name,
            "model_revision": self.config.model_revision,
            "dimension": self.config.embedding_dim,
            "strategy": self.config.chunking_strategy,
            "is_staging": True
        }

        # Étape 2 : Construction et validation dans la collection temporaire
        staging_coll = self.client.create_collection(name=staging_name, metadata=staging_metadata)
        cleaned_metas = [self.clean_metadata(m) for m in metadatas]

        staging_coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned_metas
        )

        staging_count = staging_coll.count()
        if staging_count != len(ids):
            self.client.delete_collection(staging_name)
            raise RuntimeError(
                f"Échec de validation de la collection Staging : {staging_count} indexés "
                f"pour {len(ids)} attendus. Ancienne collection préservée."
            )

        # Étape 3 : Remplacement sécurisé de la collection cible
        try:
            if target_name in [c.name for c in self.client.list_collections()]:
                self.client.delete_collection(target_name)

            new_target = self.client.create_collection(
                name=target_name,
                metadata={
                    "hnsw:space": self.config.distance_metric,
                    "model_name": self.config.model_name,
                    "model_revision": self.config.model_revision,
                    "dimension": self.config.embedding_dim,
                    "strategy": self.config.chunking_strategy
                }
            )

            # Étape 4 : Copie validée
            new_target.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=cleaned_metas
            )

            final_count = new_target.count()
            self._collection = new_target

            # Étape 5 : Nettoyage de la collection Staging après succès absolu
            self.client.delete_collection(staging_name)
            
            logger.info(f"Reconstruction réussie de '{target_name}' : {final_count} chunks indexés.")
            return final_count, "rebuild_success"

        except Exception as e:
            logger.error(
                f"Incident lors de la bascule vers '{target_name}' : {e}. "
                f"La collection Staging '{staging_name}' est conservée pour récupération d'urgence."
            )
            raise RuntimeError(f"Erreur de reconstruction : {e}")

    def count(self) -> int:
        """Retourne le nombre exact de chunks présents dans la collection cible."""
        return self.collection.count()

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un chunk individuel par son chunk_id pour contrôle d'intégrité."""
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas", "embeddings"])
        if not result or not result["ids"]:
            return None
        return {
            "chunk_id": result["ids"][0],
            "document": result["documents"][0] if result["documents"] else None,
            "metadata": result["metadatas"][0] if result["metadatas"] else None,
            "embedding": result["embeddings"][0] if result.get("embeddings") is not None else None
        }

    def list_collections(self) -> List[str]:
        """Liste l'ensemble des collections existantes dans le store persistant."""
        return [c.name for c in self.client.list_collections()]

