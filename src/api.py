"""Point d'entrée de l'API RAG sécurisée (FastAPI) connectée à Ollama et au Retriever RBAC."""
import os
import re
import time
import uuid
import requests
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.retrieval.models import UserContext, ClearanceLevel
from src.generation.models import RAGResponse
from src.generation.rag_pipeline import RAGPipeline
from src.monitoring.structured_logger import get_structured_logger

# Chargement automatique des variables d'environnement (.env)
load_dotenv()

# Expression régulière pour la validation stricte et l'assainissement de X-Request-ID
SAFE_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


class UTF8JSONResponse(JSONResponse):
    """Classe de réponse JSON garantissant l'en-tête explicite charset=utf-8 pour Windows PowerShell."""
    media_type = "application/json; charset=utf-8"


app = FastAPI(
    title="Système RAG Monolithique Sécurisé (RBAC) - DataNova Technologies",
    description="API RAG multi-sources avec pre-filtering sémantique sur métadonnées et génération locale Ollama (Llama 3.2).",
    version="1.0.0",
    default_response_class=UTF8JSONResponse
)

# Initialisation du journaliseur structuré minimal (Phase 9)
structured_logger = get_structured_logger()


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """
    Middleware d'observabilité HTTP (Phase 9) :
      - Valide, assainit ou génère l'identifiant unique de corrélation X-Request-ID.
      - Mesure avec précision la durée totale HTTP réelle de traitement via time.perf_counter().
      - Constitue l'unique source d'enregistrement des journaux d'accès pour chaque requête.
      - Journalise de façon structurée (sans fuite de données confidentielles) toutes les requêtes.
    """
    # Validation et assainissement strict de l'identifiant de corrélation client
    raw_client_request_id = request.headers.get("X-Request-ID")
    if raw_client_request_id and SAFE_REQUEST_ID_REGEX.match(raw_client_request_id):
        request_id = raw_client_request_id
    else:
        request_id = str(uuid.uuid4())

    request.state.request_id = request_id
    request.state.retrieval_time_ms = None
    request.state.generation_time_ms = None
    request.state.pipeline_time_ms = None
    request.state.has_sufficient_context = None
    request.state.resolved_clearance_level = None
    request.state.error_type = None

    start_time = time.perf_counter()
    error_type: Optional[str] = None
    response = None

    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        error_type = type(exc).__name__
        request.state.error_type = error_type
        raise
    finally:
        total_duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        status_code = response.status_code if response is not None else 500

        # Unique enregistrement de journal par requête (source de vérité : le middleware)
        if request.url.path == "/query":
            structured_logger.log_query_request(
                request_id=request_id,
                endpoint=request.url.path,
                method=request.method,
                status_code=status_code,
                total_duration_ms=total_duration_ms,
                retrieval_time_ms=getattr(request.state, "retrieval_time_ms", None),
                generation_time_ms=getattr(request.state, "generation_time_ms", None),
                pipeline_time_ms=getattr(request.state, "pipeline_time_ms", None),
                has_sufficient_context=getattr(request.state, "has_sufficient_context", None),
                resolved_clearance_level=getattr(request.state, "resolved_clearance_level", None),
                error_type=getattr(request.state, "error_type", None) or error_type
            )
        else:
            structured_logger.log_generic_request(
                request_id=request_id,
                endpoint=request.url.path,
                method=request.method,
                status_code=status_code,
                total_duration_ms=total_duration_ms,
                error_type=getattr(request.state, "error_type", None) or error_type
            )

        if response is not None:
            response.headers["X-Request-ID"] = request_id

# Configuration de l'environnement
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Registre d'authentification simulé côté serveur (Mock IAM)
# Documente explicitement la simulation de l'identité en l'absence de passerelle SSO/OAuth2 externe
SIMULATED_USER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "alice": {"roles": ["data_engineer", "employee"], "department": "Data Engineering"},
    "bob": {"roles": ["guest"], "department": "External Partner"},
    "charlie": {"roles": ["security_admin", "ciso"], "department": "Cybersecurity"},
    "guest": {"roles": ["guest"], "department": "Public"},
    "anonymous": {"roles": ["guest"], "department": "Public"}
}

# Instance singleton paresseuse du pipeline RAG
_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Initialisation paresseuse du RAGPipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question posée par l'utilisateur")
    user_clearance: int = Field(
        ...,
        ge=1,
        le=3,
        description="Niveau d'accréditation demandé (1: Public, 2: Interne, 3: Restreint)"
    )
    user_id: Optional[str] = Field(None, description="Identifiant utilisateur pour la résolution de profil côté serveur")
    top_k: Optional[int] = Field(4, ge=1, le=50, description="Nombre de documents demandés")


class ChunkProvenance(BaseModel):
    id: str
    document_id: Optional[str] = None
    source: str
    date_version: Optional[str] = "1.0.0"
    doc_type: str
    sensitivity_level: int
    reference: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    llm_model: str
    sources_retrieved: List[ChunkProvenance]
    chunks_filtered_out: int = 0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    has_sufficient_context: bool = True


def resolve_server_user_context(request: QueryRequest) -> UserContext:
    """
    Résolution sécurisée de l'identité et du niveau d'accréditation côté serveur.
    
    Garanties de sécurité strictes (Zero Trust & Moindre Privilège) :
      - Ne fait JAMAIS confiance au paramètre client `user_clearance` comme source d'autorisation.
      - N'accepte JAMAIS de liste de rôles fournie par le client.
      - L'autorisation est strictement dérivée des rôles certifiés dans SIMULATED_USER_REGISTRY.
      - Tout utilisateur inconnu ou non renseigné reçoit le niveau de moindre privilège (Public / guest).
      - Construction du contexte via l'interface officielle UserContext.from_roles().
    """
    user_id = request.user_id.strip() if request.user_id and request.user_id.strip() else "anonymous"

    # Cas 1 : Utilisateur reconnu dans le registre certifié côté serveur
    if user_id in SIMULATED_USER_REGISTRY:
        profile = SIMULATED_USER_REGISTRY[user_id]
        trusted_roles = profile.get("roles", ["guest"])
        department = profile.get("department")
        return UserContext.from_roles(
            user_id=user_id,
            roles=trusted_roles,
            department=department
        )

    # Cas 2 : Utilisateur non répertorié -> Traitement de moindre privilège (Public / guest)
    return UserContext.from_roles(
        user_id=user_id,
        roles=["guest"],
        department="Public"
    )


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Endpoint de santé pour diagnostic d'Ollama et de l'environnement."""
    ollama_online = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=2)
        ollama_online = (r.status_code == 200)
    except Exception:
        ollama_online = False

    return {
        "status": "healthy",
        "service": "rag-monolith-api",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
        "ollama_connected": ollama_online
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest, raw_request: Request) -> QueryResponse:
    """
    Endpoint RAG sécurisé de bout en bout avec observabilité minimale (Phase 9) :
      1. Résolution de l'accréditation RBAC côté backend.
      2. Pre-filtering natif ChromaDB et validation défensive (Zero Trust).
      3. Inférence locale Ollama (Llama 3.2).
      4. Restitution de la réponse avec citations réelles de provenance.
      5. Transmission de la télémétrie RAG au middleware via raw_request.state.
    """
    user_context = resolve_server_user_context(request)
    raw_request.state.resolved_clearance_level = user_context.clearance_level
    pipeline = get_pipeline()

    try:
        rag_response: RAGResponse = pipeline.run(
            query=request.query,
            user_context=user_context,
            top_k=request.top_k
        )
        sources_provenance = [
            ChunkProvenance(
                id=s.chunk_id,
                document_id=s.document_id,
                source=s.source,
                date_version="1.0.0",
                doc_type=s.doc_type,
                sensitivity_level=s.sensitivity_level,
                reference=s.reference
            )
            for s in rag_response.sources
        ]

        # Transmission des métriques RAG au middleware (unique source d'enregistrement)
        raw_request.state.retrieval_time_ms = rag_response.retrieval_time_ms
        if rag_response.chunks_retrieved_count > 0 and rag_response.generation_time_ms > 0.0:
            raw_request.state.generation_time_ms = rag_response.generation_time_ms
        else:
            raw_request.state.generation_time_ms = None
        raw_request.state.pipeline_time_ms = rag_response.total_time_ms
        raw_request.state.has_sufficient_context = rag_response.has_sufficient_context
        raw_request.state.error_type = None

        return QueryResponse(
            query=rag_response.query,
            answer=rag_response.answer,
            llm_model=rag_response.model_name,
            sources_retrieved=sources_provenance,
            chunks_filtered_out=0,
            retrieval_time_ms=rag_response.retrieval_time_ms,
            generation_time_ms=rag_response.generation_time_ms,
            total_time_ms=rag_response.total_time_ms,
            has_sufficient_context=rag_response.has_sufficient_context
        )
    except Exception as e:
        # En cas d'indisponibilité du serveur Ollama en environnement de test hors-ligne,
        # repli gracieux extrayant les sources autorisées depuis le Retriever ChromaDB
        try:
            retrieval_res = pipeline.retriever.retrieve(
                query=request.query,
                user_context=user_context,
                top_k=request.top_k
            )
            fallback_sources = [
                ChunkProvenance(
                    id=c.chunk_id,
                    document_id=c.metadata.get("document_id"),
                    source=c.metadata.get("source", "unknown"),
                    date_version=c.metadata.get("date_version", "1.0.0"),
                    doc_type=c.metadata.get("doc_type", "document"),
                    sensitivity_level=c.sensitivity_level,
                    reference=c.metadata.get("reference")
                )
                for c in retrieval_res.chunks
            ]

            # Transmission des métriques RAG en mode dégradé (sans génération LLM effective)
            raw_request.state.retrieval_time_ms = retrieval_res.execution_time_ms
            raw_request.state.generation_time_ms = None
            raw_request.state.pipeline_time_ms = retrieval_res.execution_time_ms
            raw_request.state.has_sufficient_context = False
            raw_request.state.error_type = "OllamaOfflineDegraded"

            return QueryResponse(
                query=request.query,
                answer=f"Génération LLM indisponible (Ollama hors ligne). {len(fallback_sources)} sources autorisées identifiées.",
                llm_model=OLLAMA_MODEL,
                sources_retrieved=fallback_sources,
                chunks_filtered_out=0,
                retrieval_time_ms=retrieval_res.execution_time_ms,
                generation_time_ms=0.0,
                total_time_ms=retrieval_res.execution_time_ms,
                has_sufficient_context=False
            )
        except Exception as fatal_e:
            raw_request.state.retrieval_time_ms = None
            raw_request.state.generation_time_ms = None
            raw_request.state.pipeline_time_ms = None
            raw_request.state.has_sufficient_context = False
            raw_request.state.error_type = type(fatal_e).__name__
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur d'exécution du pipeline RAG : {str(fatal_e)}"
            )
