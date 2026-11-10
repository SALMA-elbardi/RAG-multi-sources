"""Orchestrateur du pipeline RAG reliant le Retriever RBAC, le PromptBuilder et Ollama."""
import logging
import time
from typing import Optional, List, Dict, Any

from src.retrieval.models import UserContext, ClearanceLevel
from src.retrieval.semantic_retriever import SemanticRetriever
from src.generation.models import (
    LLMConfig,
    DocumentCitation,
    RAGResponse,
    RAGPipelineError,
    extract_citation_from_chunk
)
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_client import OllamaClient

logger = logging.getLogger(__name__)


# Marqueurs sémantiques indiquant une insuffisance d'information détectée par le LLM
INSUFFICIENT_CONTEXT_MARKERS = [
    "ne contiennent pas d'informations suffisantes",
    "ne permettent pas de répondre",
    "informations insuffisantes",
    "aucun document ne mentionne",
    "aucun document ne précise",
    "aucun document fourni ne permet",
    "pas suffisamment d'informations",
    "ne contient pas d'information"
]


class RAGPipeline:
    """
    Pipeline RAG monolithique sécurisé de DataNova Technologies.
    
    Orchestration :
      1. Validation de la requête et de l'identité utilisateur côté serveur.
      2. Interrogation du Retriever RBAC (Pre-filtering + Default-Deny).
      3. Contrôle de vacuité : Repli immédiat sans appel LLM si 0 chunk autorisé.
      4. Extraction certifiée des métadonnées de citations (indépendamment du LLM).
      5. Construction du prompt sécurisé avec balisage anti-injection.
      6. Inférence locale Ollama (Llama 3.2).
      7. Évaluation de la complétude documentaire (`has_sufficient_context`).
      8. Consolidation de la RAGResponse avec mesure indépendante des latences.
    """

    def __init__(
        self,
        retriever: Optional[SemanticRetriever] = None,
        llm_client: Optional[OllamaClient] = None,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        self.retriever = retriever or SemanticRetriever()
        self.llm_client = llm_client or OllamaClient()
        self.prompt_builder = prompt_builder or PromptBuilder()

    @staticmethod
    def _evaluate_context_sufficiency(answer: str) -> bool:
        """
        Évalue si le contexte documentaire était suffisant pour répondre.
        
        Méthode d'évaluation :
          Analyse textuelle déterministe recherchant les formules explicites
          d'insuffisance documentaire exigées par la consigne système.
          Retourne False si le LLM signale un manque de faits, True sinon.
        """
        if not answer:
            return False

        normalized = answer.lower()
        for marker in INSUFFICIENT_CONTEXT_MARKERS:
            if marker in normalized:
                return False

        return True

    def run(
        self,
        query: str,
        user_context: UserContext,
        top_k: Optional[int] = None,
        min_similarity_score: Optional[float] = None
    ) -> RAGResponse:
        """
        Exécute la boucle RAG complète de manière synchrone et sécurisée.
        """
        pipeline_start = time.perf_counter()

        # 1. Validation de la requête
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("La requête utilisateur ne peut pas être vide ou composée uniquement d'espaces.")
        clean_query = query.strip()

        # 2. Validation de l'identité utilisateur
        if not isinstance(user_context, UserContext):
            raise TypeError(
                f"Le contexte utilisateur doit être une instance de UserContext (reçu '{type(user_context).__name__}')."
            )

        # 3. Étape de Retrieval sécurisé
        try:
            retrieval_result = self.retriever.retrieve(
                query=clean_query,
                user_context=user_context,
                top_k=top_k,
                min_similarity_score=min_similarity_score
            )
        except Exception as e:
            raise RAGPipelineError(f"Échec lors de l'étape de recherche documentaire (Retrieval) : {e}")

        retrieval_time_ms = retrieval_result.execution_time_ms
        authorized_chunks = retrieval_result.chunks

        # 4. Situation A : Aucun chunk autorisé récupéré -> Pas d'appel Ollama
        if not authorized_chunks:
            total_time_ms = round((time.perf_counter() - pipeline_start) * 1000.0, 2)
            logger.info(
                f"RAGPipeline : 0 chunk autorisé pour la requête '{clean_query}' "
                f"[user_id='{user_context.user_id}', clearance={user_context.clearance_level}]. "
                f"Génération annulée, retour de la réponse de repli."
            )
            return RAGResponse(
                query=clean_query,
                answer="Les documents autorisés disponibles ne permettent pas de répondre à cette question (aucun document pertinent accessible pour votre niveau d'accréditation).",
                user_id=user_context.user_id,
                clearance_level=user_context.clearance_level,
                sources=[],
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=0.0,
                total_time_ms=total_time_ms,
                model_name=self.llm_client.config.model_name,
                has_sufficient_context=False,
                chunks_retrieved_count=0
            )

        # 5. Extraction certifiée des citations côté serveur
        citations: List[DocumentCitation] = [
            extract_citation_from_chunk(c) for c in authorized_chunks
        ]

        # 6. Construction du prompt sécurisé
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            query=clean_query,
            chunks=authorized_chunks
        )

        # 7. Situation B : Génération via le LLM local
        try:
            generated_answer, generation_time_ms = self.llm_client.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )
        except Exception as e:
            raise RAGPipelineError(f"Échec lors de la génération avec le modèle LLM local : {e}")

        # 8. Évaluation de la complétude du contexte
        has_sufficient_context = self._evaluate_context_sufficiency(generated_answer)

        total_time_ms = round((time.perf_counter() - pipeline_start) * 1000.0, 2)

        return RAGResponse(
            query=clean_query,
            answer=generated_answer.strip(),
            user_id=user_context.user_id,
            clearance_level=user_context.clearance_level,
            sources=citations,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            model_name=self.llm_client.config.model_name,
            has_sufficient_context=has_sufficient_context,
            chunks_retrieved_count=len(authorized_chunks)
        )


# ==============================================================================
# SCRIPT DE DÉMONSTRATION CLI DU PIPELINE DE BOUT EN BOUT
# ==============================================================================

if __name__ == "__main__":
    import json

    print("\n" + "=" * 80)
    print("DATANOVA TECHNOLOGIES — PIPELINE RAG COMPLET (LLM LOCAL & SÉCURITÉ RBAC)")
    print("=" * 80)

    # Initialisation du pipeline
    pipeline = RAGPipeline()
    print(f"\n[*] Pipeline initialisé avec le modèle LLM local : '{pipeline.llm_client.config.model_name}'")
    print(f"[*] Vérification de l'accès Ollama ({pipeline.llm_client.config.base_url}) : ", end="")
    is_online = pipeline.llm_client.check_health()
    print("EN LIGNE\n" if is_online else "HORS LIGNE (Assurez-vous que 'ollama serve' est lancé)\n")

    test_scenarios = [
        (
            "Scénario 1 (Niveau 1 - Public)",
            "Quels sont les datasets disponibles pour la télémétrie IoT ?",
            UserContext.from_roles("user_public", ["guest"])
        ),
        (
            "Scénario 2 (Niveau 2 - Interne)",
            "Quelles sont les architectures recommandées pour le Lakehouse DataNova ?",
            UserContext.from_roles("user_engineer", ["data_engineer"])
        ),
        (
            "Scénario 3 (Niveau 1 - Question confidentielle bloquée)",
            "Quels sont les montants des salaires et primes confidentiels ?",
            UserContext.from_roles("user_guest", ["guest"])
        )
    ]

    for title, q, ctx in test_scenarios:
        print("-" * 80)
        print(f"Test : {title}")
        print(f"Utilisateur : {ctx.user_id} (Clearance : Niveau {ctx.clearance_level})")
        print(f"Question    : \"{q}\"")

        try:
            resp = pipeline.run(query=q, user_context=ctx, top_k=3)
            print(f"\nRéponse générée :\n{resp.answer}\n")
            print(f"Sources certifiées ({len(resp.sources)}) :")
            for s in resp.sources:
                ref_str = f" | {s.reference}" if s.reference else ""
                print(f"  - [{s.document_id}] {s.source} (Niveau: {s.sensitivity_level}{ref_str})")
            print(f"\nLatences : Retrieval={resp.retrieval_time_ms} ms | Génération={resp.generation_time_ms} ms | Total={resp.total_time_ms} ms")
            print(f"Contexte suffisant : {resp.has_sufficient_context}")
        except Exception as e:
            print(f"[ERREUR] Échec de la génération : {e}")
        print()

    print("=" * 80 + "\n")

