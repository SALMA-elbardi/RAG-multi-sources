"""Tests des endpoints FastAPI et validation des contraintes d'API / RBAC."""
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check_endpoint():
    """Vérifie le bon fonctionnement du healthcheck de l'API."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "rag-monolith-api"
    assert "ollama_connected" in data
    assert "mlflow_uri" in data

def test_query_endpoint_valid_rbac_level():
    """Vérifie qu'une requête avec une accréditation RBAC valide (1..3) est acceptée."""
    payload = {
        "query": "Quelles sont les consignes de sécurité ?",
        "user_clearance": 2
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert "llm_model" in data
    assert "sources_retrieved" in data
    assert isinstance(data["sources_retrieved"], list)

def test_query_endpoint_invalid_rbac_level_rejected():
    """Vérifie le rejet strict des niveaux RBAC hors bornes (< 1 ou > 3)."""
    # Niveau 0 (invalide)
    res_low = client.post("/query", json={"query": "test", "user_clearance": 0})
    assert res_low.status_code == 422

    # Niveau 4 (invalide)
    res_high = client.post("/query", json={"query": "test", "user_clearance": 4})
    assert res_high.status_code == 422

def test_query_endpoint_missing_fields_rejected():
    """Vérifie le rejet des payloads incomplets."""
    # Requête sans query
    res = client.post("/query", json={"user_clearance": 1})
    assert res.status_code == 422

    # Requête sans user_clearance
    res2 = client.post("/query", json={"query": "test"})
    assert res2.status_code == 422

