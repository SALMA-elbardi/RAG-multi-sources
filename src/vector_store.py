"""Gestionnaire du Vector Store monolithique ChromaDB avec filtrage sémantique RBAC."""
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

from src.schema import DocumentChunk, ChunkMetadata


class MonolithicVectorStore:
    """
    Gestionnaire d'index vectoriel unique (ChromaDB) pour RAG monolithique.
    Contrainte stricte : le contrôle d'accès (RBAC) s'effectue avant/pendant
    la recherche sémantique via une clause de pre-filtering vectorielle.
    """

    COLLECTION_NAME = "rag_monolith_documents"

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_store")
        
        # Initialisation du client persistant ChromaDB
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialisation des embeddings (exécutés localement)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Récupération ou création de la collection unique
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Index vectoriel unique multi-sources avec pre-filtering RBAC"}
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Indexe une liste de chunks avec leurs métadonnées normalisées."""
        if not chunks:
            return 0

        ids = [c.metadata.id for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = [c.metadata.to_chroma_metadata() for c in chunks]
        
        # Calcul des embeddings
        embeddings = self.embeddings.embed_documents(texts)

        # Upsert pour éviter les doublons lors des re-runs DVC
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        return len(chunks)

    def search_with_rbac(
        self,
        query: str,
        user_clearance: int,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Recherche sémantique sécurisée par pre-filtering RBAC strict.
        
        La clause `where` est appliquée au niveau du moteur vectoriel HNSW.
        Un chunk de sensibilité > user_clearance ne peut PHYSIQUEMENT PAS
        être sélectionné ni parvenir au contexte du LLM.
        """
        query_embedding = self.embeddings.embed_query(query)

        # Pre-filtering vectoriel strict : sensitivity_level <= user_clearance
        # Exemple : si user_clearance = 2, seuls les niveaux 1 et 2 sont consultés
        where_filter = {
            "sensitivity_level": {"$lte": int(user_clearance)}
        }

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved: List[Dict[str, Any]] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return retrieved

        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            retrieved.append({
                "content": doc,
                "metadata": meta,
                "distance": dist
            })

        return retrieved

    def count(self) -> int:
        """Retourne le nombre total de chunks dans l'index."""
        return self.collection.count()

    def reset(self) -> None:
        """Supprime la collection pour les tests ou ré-indexations complètes."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Index vectoriel unique multi-sources avec pre-filtering RBAC"}
        )

