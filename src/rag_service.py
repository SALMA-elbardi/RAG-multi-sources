"""Service d'orchestration RAG monolithique sécurisé connecté à Ollama."""
import os
from typing import List, Dict, Any, Optional
import requests
from langchain_core.prompts import PromptTemplate

from src.schema import DocumentChunk, ChunkMetadata
from src.vector_store import MonolithicVectorStore


SYSTEM_PROMPT_TEMPLATE = """Tu es un assistant IA documentaire hautement sécurisé respectant le contrôle d'accès RBAC.
Tu réponds à la question de l'utilisateur UNIQUEMENT en te basant sur le contexte documentaire autorisé fourni ci-dessous.
Si le contexte fourni ne contient pas l'information ou si l'utilisateur ne dispose pas du niveau de sensibilité requis, indique clairement que l'information n'est pas accessible avec ses droits d'accès.

Ne divulgue jamais d'informations en dehors de ce contexte.

==================================================
CONTEXTE AUTORISÉ (Niveau d'accréditation utilisateur : {user_clearance}) :
==================================================
{context}
==================================================

QUESTION DE L'UTILISATEUR :
{query}

RÉPONSE CLAIRE ET STRUCTURÉE :"""


class RAGService:
    """
    Orchestrateur du flux RAG monolithique.
    Garantit la contrainte stricte de sécurité : pre-filtering sémantique au retrieval
    et validation défensive avant injection dans le contexte du LLM Ollama.
    """

    def __init__(
        self,
        vector_store: Optional[MonolithicVectorStore] = None,
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None
    ):
        self.vector_store = vector_store or MonolithicVectorStore()
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.prompt_template = PromptTemplate(
            template=SYSTEM_PROMPT_TEMPLATE,
            input_variables=["user_clearance", "context", "query"]
        )

    def format_context(self, retrieved_chunks: List[Dict[str, Any]], user_clearance: int) -> str:
        """
        Formate les chunks récupérés en contexte textuel enrichi de métadonnées de provenance.
        Défense en profondeur : vérifie qu'aucun chunk > user_clearance ne franchit ce stade.
        """
        if not retrieved_chunks:
            return "Aucun document autorisé disponible pour cette requête."

        formatted_blocks = []
        for idx, item in enumerate(retrieved_chunks, start=1):
            meta = item.get("metadata", {})
            chunk_sensitivity = int(meta.get("sensitivity_level", 3))

            # Garde-fou strict de sécurité RBAC (Défense en profondeur)
            if chunk_sensitivity > user_clearance:
                continue

            source = meta.get("source", "Inconnue")
            doc_type = meta.get("doc_type", "N/A")
            version = meta.get("date_version", "N/A")
            chunk_id = meta.get("id", f"chunk_{idx}")
            content = item.get("content", "").strip()

            block = (
                f"[Source {idx} | ID: {chunk_id} | Document: {source} ({doc_type}) | "
                f"Version: {version} | Sensibilité: {chunk_sensitivity}]\n"
                f"{content}"
            )
            formatted_blocks.append(block)

        return "\n\n---\n\n".join(formatted_blocks)

    def call_ollama(self, prompt: str) -> str:
        """Appelle l'API Ollama locale pour générer la réponse."""
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Faible température pour fidélité au contexte
                "top_p": 0.9
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            else:
                return f"[Erreur Ollama HTTP {response.status_code}]: Impossible de générer la réponse."
        except requests.exceptions.ConnectionError:
            return (
                f"[Avertissement Ollama]: Impossible de joindre Ollama sur {self.ollama_base_url}. "
                "Vérifiez qu'Ollama est démarré avec 'ollama serve'."
            )
        except Exception as e:
            return f"[Erreur Inférence]: {str(e)}"

    def query(self, query: str, user_clearance: int, top_k: int = 4) -> Dict[str, Any]:
        """
        Exécute le pipeline RAG complet et sécurisé :
        1. Pre-filtering sémantique au niveau du Vector Store (ChromaDB)
        2. Validation défensive de non-dépassement d'accréditation
        3. Formatage du prompt sécurisé avec traçabilité de provenance
        4. Inférence avec le LLM Ollama
        """
        # 1. Retrieval avec pre-filtering RBAC strict
        retrieved_chunks = self.vector_store.search_with_rbac(
            query=query,
            user_clearance=user_clearance,
            top_k=top_k
        )

        # 2. Formatage du contexte
        context_text = self.format_context(retrieved_chunks, user_clearance)

        # 3. Construction du prompt
        prompt = self.prompt_template.format(
            user_clearance=user_clearance,
            context=context_text,
            query=query
        )

        # 4. Inférence LLM
        answer = self.call_ollama(prompt)

        # Formatage des sources autorisées retournées au client
        sources = [
            {
                "id": c["metadata"].get("id", ""),
                "source": c["metadata"].get("source", ""),
                "date_version": c["metadata"].get("date_version", ""),
                "doc_type": c["metadata"].get("doc_type", ""),
                "sensitivity_level": int(c["metadata"].get("sensitivity_level", 1))
            }
            for c in retrieved_chunks
        ]

        return {
            "query": query,
            "answer": answer,
            "llm_model": self.ollama_model,
            "sources_retrieved": sources,
            "context_prompt_chars": len(context_text)
        }

