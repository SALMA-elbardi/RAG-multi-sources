"""Point d'entrée de l'API RAG sécurisée (FastAPI) connectée à Ollama."""
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import requests
from dotenv import load_dotenv

# Chargement automatique des variables d'environnement (.env)
load_dotenv()

app = FastAPI(
    title="Système RAG Monolithique Sécurisé (RBAC) - Ollama Local",
    description="API RAG multi-sources avec pre-filtering sémantique sur métadonnées et inférence locale Ollama.",
    version="1.0.0"
)

# Configuration de l'environnement
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Question posée par l'utilisateur")
    user_clearance: int = Field(
        ...,
        ge=1,
        le=3,
        description="Niveau d'accréditation RBAC (1: Public, 2: Interne, 3: Restreint)"
    )

class ChunkProvenance(BaseModel):
    id: str
    source: str
    date_version: str
    doc_type: str
    sensitivity_level: int

class QueryResponse(BaseModel):
    query: str
    answer: str
    llm_model: str
    sources_retrieved: List[ChunkProvenance]
    chunks_filtered_out: int = 0

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Endpoint de santé pour Docker et diagnostic d'Ollama."""
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
def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Endpoint RAG sécurisé avec pre-filtering RBAC strict avant l'inférence Ollama.
    Aucun chunk d'un niveau > user_clearance ne sera injecté dans le prompt Ollama.
    """
    return QueryResponse(
        query=request.query,
        answer=f"API initialisée avec le modèle local Ollama '{OLLAMA_MODEL}'. Le module de retrieval filtré sera connecté à l'étape suivante.",
        llm_model=OLLAMA_MODEL,
        sources_retrieved=[],
        chunks_filtered_out=0
    )
