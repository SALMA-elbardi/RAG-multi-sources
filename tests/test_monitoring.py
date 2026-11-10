"""Tests automatisés pour le monitoring minimal et l'observabilité du RAG (Phase 9)."""
import json
import uuid
import re
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.api import app
from src.generation.models import RAGResponse
from src.monitoring.structured_logger import StructuredLogger
from src.monitoring.metrics_analyzer import MetricsAnalyzer, calculate_percentile


@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """Fixture fournissant un chemin temporaire pour les journaux de test."""
    return tmp_path / "test_api_access.jsonl"


@pytest.fixture(autouse=True)
def mock_health_requests(monkeypatch):
    """Accélère instantanément les appels aux endpoints de diagnostic /health."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"version": "0.3.14"}
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_resp)


def test_percentile_calculation():
    """1. Vérifie le calcul exact du percentile (P95) selon la méthode Nearest-Rank."""
    # Liste vide
    assert calculate_percentile([], 95.0) == 0.0

    # Élément unique
    assert calculate_percentile([42.5], 95.0) == 42.5

    # Échantillon de 10 valeurs [10, 20, 30, ..., 100]
    sample = [float(i * 10) for i in range(1, 11)]
    # N=10, 95% -> rank = ceil(0.95 * 10) = 10 -> index 9 -> 100.0
    assert calculate_percentile(sample, 95.0) == 100.0

    # Échantillon de 100 valeurs [1..100]
    sample_100 = [float(i) for i in range(1, 101)]
    # N=100, 95% -> rank = 95 -> index 94 -> 95.0
    assert calculate_percentile(sample_100, 95.0) == 95.0


def test_structured_logger_write_and_format(temp_log_file: Path):
    """2. Vérifie la conformité du format JSONL et l'absence de fuite de données sensibles."""
    logger = StructuredLogger(log_path=temp_log_file)

    # 1. Journalisation d'une requête RAG avec génération effective
    logger.log_query_request(
        request_id="req-12345",
        endpoint="/query",
        method="POST",
        status_code=200,
        total_duration_ms=150.25,
        retrieval_time_ms=12.5,
        generation_time_ms=130.0,
        pipeline_time_ms=142.5,
        has_sufficient_context=True,
        resolved_clearance_level=2,
        error_type=None
    )

    # 2. Journalisation d'une requête générique
    logger.log_generic_request(
        request_id="req-67890",
        endpoint="/health",
        method="GET",
        status_code=200,
        total_duration_ms=5.12
    )

    assert temp_log_file.exists()
    lines = temp_log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    # Validation du premier événement
    ev1 = json.loads(lines[0])
    assert ev1["request_id"] == "req-12345"
    assert ev1["endpoint"] == "/query"
    assert ev1["method"] == "POST"
    assert ev1["status_code"] == 200
    assert ev1["total_duration_ms"] == 150.25
    assert ev1["retrieval_time_ms"] == 12.5
    assert ev1["generation_time_ms"] == 130.0
    assert ev1["pipeline_time_ms"] == 142.5
    assert ev1["has_sufficient_context"] is True
    assert ev1["resolved_clearance_level"] == 2
    assert ev1["error_type"] is None
    assert "timestamp" in ev1

    # RÈGLE DE SÉCURITÉ STRICTE : Aucun champ de contenu textuel (prompt, réponse, document) n'est présent
    for forbidden_key in ("prompt", "answer", "query", "document", "chunk_content", "user_id"):
        assert forbidden_key not in ev1


def test_metrics_analyzer_computations(temp_log_file: Path):
    """3. Vérifie l'exactitude des calculs statistiques (taux d'erreur, moyennes, P95, contextes)."""
    logger = StructuredLogger(log_path=temp_log_file)

    # Simulation de 4 événements : 3 succès (200) et 1 échec client (422)
    logger.log_query_request(
        request_id="req-1",
        endpoint="/query",
        method="POST",
        status_code=200,
        total_duration_ms=100.0,
        retrieval_time_ms=10.0,
        generation_time_ms=80.0,
        pipeline_time_ms=90.0,
        has_sufficient_context=True,
        resolved_clearance_level=1
    )
    logger.log_query_request(
        request_id="req-2",
        endpoint="/query",
        method="POST",
        status_code=200,
        total_duration_ms=200.0,
        retrieval_time_ms=20.0,
        generation_time_ms=160.0,
        pipeline_time_ms=180.0,
        has_sufficient_context=True,
        resolved_clearance_level=2
    )
    logger.log_query_request(
        request_id="req-3",
        endpoint="/query",
        method="POST",
        status_code=200,
        total_duration_ms=300.0,
        retrieval_time_ms=30.0,
        generation_time_ms=240.0,
        pipeline_time_ms=270.0,
        has_sufficient_context=False,
        resolved_clearance_level=1
    )
    logger.log_generic_request(
        request_id="req-4",
        endpoint="/query",
        method="POST",
        status_code=422,
        total_duration_ms=10.0,
        error_type="RequestValidationError"
    )

    analyzer = MetricsAnalyzer(log_path=temp_log_file)
    m = analyzer.compute_metrics()

    # Volumétrie et taux d'erreur
    assert m["total_requests"] == 4
    assert m["successful_requests"] == 3
    assert m["failed_requests"] == 1
    assert m["error_rate_pct"] == 25.0

    # Latences HTTP totales (10, 100, 200, 300) -> moyenne = 152.5
    assert m["mean_total_duration_ms"] == 152.5
    assert m["p95_total_duration_ms"] == 300.0

    # Latences de retrieval (10, 20, 30) -> moyenne = 20.0
    assert m["mean_retrieval_time_ms"] == 20.0
    assert m["p95_retrieval_time_ms"] == 30.0

    # Latences de génération (80, 160, 240) -> moyenne = 160.0
    assert m["mean_generation_time_ms"] == 160.0
    assert m["generation_metrics_available"] is True
    assert m["generation_count"] == 3

    # Complétude du contexte
    assert m["sufficient_context_count"] == 2
    assert m["insufficient_context_count"] == 1

    # Répartition RBAC
    assert m["requests_by_clearance"]["Niveau 1 (Public)"] == 2
    assert m["requests_by_clearance"]["Niveau 2 (Interne)"] == 1

    # Statuts HTTP et erreurs
    assert m["requests_by_status"][200] == 3
    assert m["requests_by_status"][422] == 1
    assert m["error_types"]["RequestValidationError"] == 1


def test_metrics_analyzer_generation_none_and_unavailable(temp_log_file: Path):
    """4. Vérifie l'exclusion des générations non exécutées (None) et le message explicite d'indisponibilité."""
    logger = StructuredLogger(log_path=temp_log_file)

    # Scénario A : Une génération effective (120ms) et une sans génération (Situation A / None)
    logger.log_query_request(
        request_id="req-gen-1",
        endpoint="/query",
        method="POST",
        status_code=200,
        total_duration_ms=130.0,
        retrieval_time_ms=10.0,
        generation_time_ms=120.0,
        pipeline_time_ms=130.0,
        has_sufficient_context=True,
        resolved_clearance_level=1
    )
    logger.log_query_request(
        request_id="req-no-gen-2",
        endpoint="/query",
        method="POST",
        status_code=200,
        total_duration_ms=15.0,
        retrieval_time_ms=15.0,
        generation_time_ms=None,  # Aucune génération LLM
        pipeline_time_ms=15.0,
        has_sufficient_context=False,
        resolved_clearance_level=1
    )

    analyzer = MetricsAnalyzer(log_path=temp_log_file)
    m = analyzer.compute_metrics()

    # La moyenne de génération doit être 120.0 (et NON faussée par 0.0)
    assert m["mean_generation_time_ms"] == 120.0
    assert m["p95_generation_time_ms"] == 120.0
    assert m["generation_count"] == 1
    assert m["generation_metrics_available"] is True

    # Scénario B : Journal où AUCUNE requête n'a fait l'objet d'une génération
    temp_no_gen = temp_log_file.parent / "no_gen.jsonl"
    logger_no_gen = StructuredLogger(log_path=temp_no_gen)
    logger_no_gen.log_query_request(
        request_id="req-only-fallback",
        endpoint="/query",
        method="POST",
        status_code=200,
        total_duration_ms=20.0,
        retrieval_time_ms=20.0,
        generation_time_ms=None,
        pipeline_time_ms=20.0,
        has_sufficient_context=False,
        resolved_clearance_level=1
    )

    analyzer_no_gen = MetricsAnalyzer(log_path=temp_no_gen)
    m_no_gen = analyzer_no_gen.compute_metrics()
    assert m_no_gen["mean_generation_time_ms"] is None
    assert m_no_gen["p95_generation_time_ms"] is None
    assert m_no_gen["generation_metrics_available"] is False

    report = analyzer_no_gen.generate_report()
    assert "Indisponible (aucune génération LLM effective enregistrée)" in report


def test_metrics_analyzer_empty_log(tmp_path: Path):
    """5. Vérifie le comportement propre de l'analyseur face à un journal vide ou inexistant."""
    missing_file = tmp_path / "non_existent.jsonl"
    analyzer = MetricsAnalyzer(log_path=missing_file)
    m = analyzer.compute_metrics()

    assert m["total_requests"] == 0
    assert m["error_rate_pct"] == 0.0
    assert m["mean_total_duration_ms"] == 0.0
    assert m["mean_generation_time_ms"] is None
    assert m["generation_metrics_available"] is False
    assert len(m["unavailable_metrics"]) > 0

    # Le rapport textuel ne doit pas planter et afficher l'indisponibilité de génération
    report = analyzer.generate_report()
    assert "DATANOVA TECHNOLOGIES" in report
    assert "Nombre total de requêtes    : 0" in report
    assert "Indisponible (aucune génération LLM effective enregistrée)" in report


def test_api_x_request_id_sanitization_and_propagation():
    """6. Vérifie la validation, l'assainissement strict et la propagation de X-Request-ID."""
    client = TestClient(app)

    # 1. Requête sans en-tête -> Génération automatique d'un UUIDv4 valide
    res1 = client.get("/health")
    assert res1.status_code == 200
    assert "X-Request-ID" in res1.headers
    auto_id = res1.headers["X-Request-ID"]
    uuid_obj = uuid.UUID(auto_id)
    assert uuid_obj.version == 4

    # 2. Requête avec identifiant client valide -> Préservé
    valid_id = "trace-client_2026.04-ok"
    res2 = client.get("/health", headers={"X-Request-ID": valid_id})
    assert res2.status_code == 200
    assert res2.headers["X-Request-ID"] == valid_id

    # 3. Requête avec caractères invalides (injection / ponctuation non autorisée) -> Remplacé par UUIDv4
    bad_id = "trace<script>alert(1)</script>"
    res3 = client.get("/health", headers={"X-Request-ID": bad_id})
    assert res3.status_code == 200
    sanitized_id = res3.headers["X-Request-ID"]
    assert sanitized_id != bad_id
    assert uuid.UUID(sanitized_id).version == 4

    # 4. Requête avec identifiant trop long (> 64 caractères) -> Remplacé par UUIDv4
    too_long_id = "a" * 65
    res4 = client.get("/health", headers={"X-Request-ID": too_long_id})
    assert res4.status_code == 200
    sanitized_too_long = res4.headers["X-Request-ID"]
    assert sanitized_too_long != too_long_id
    assert uuid.UUID(sanitized_too_long).version == 4

    # 5. Requête avec chaîne vide -> Remplacé par UUIDv4
    res5 = client.get("/health", headers={"X-Request-ID": ""})
    assert res5.status_code == 200
    assert uuid.UUID(res5.headers["X-Request-ID"]).version == 4


def test_api_query_single_log_and_middleware_duration(monkeypatch, tmp_path: Path):
    """7. Vérifie l'enregistrement d'un UNIQUE log par appel /query et la mesure de durée issue du middleware."""
    isolated_log = tmp_path / "single_log_test.jsonl"
    custom_logger = StructuredLogger(log_path=isolated_log)
    monkeypatch.setattr("src.api.structured_logger", custom_logger)

    # Simulation d'une exécution nominale RAG avec latences connues
    mock_response = RAGResponse(
        query="Question nominale",
        answer="Réponse certifiée",
        user_id="alice",
        clearance_level=2,
        sources=[],
        retrieval_time_ms=18.5,
        generation_time_ms=110.0,
        total_time_ms=128.5,
        model_name="llama3.2",
        has_sufficient_context=True,
        chunks_retrieved_count=2
    )
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = mock_response
    monkeypatch.setattr("src.api.get_pipeline", lambda: mock_pipeline)

    client = TestClient(app)
    response = client.post(
        "/query",
        json={"query": "Question nominale", "user_clearance": 2, "user_id": "alice"},
        headers={"X-Request-ID": "test-single-log-trace"}
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-single-log-trace"

    # Vérification stricte : UN SEUL enregistrement dans le fichier de log
    assert isolated_log.exists()
    lines = isolated_log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    event = json.loads(lines[0])
    assert event["request_id"] == "test-single-log-trace"
    assert event["endpoint"] == "/query"
    assert event["status_code"] == 200
    # La durée HTTP totale provient du middleware et est strictement positive
    assert event["total_duration_ms"] > 0
    # Les métriques RAG sont transmises par request.state
    assert event["retrieval_time_ms"] == 18.5
    assert event["generation_time_ms"] == 110.0
    assert event["pipeline_time_ms"] == 128.5
    assert event["has_sufficient_context"] is True
    assert event["resolved_clearance_level"] == 2
    assert event["error_type"] is None


def test_api_query_unperformed_generation_logs_none(monkeypatch, tmp_path: Path):
    """8. Vérifie que l'absence de génération effective enregistre generation_time_ms = null."""
    isolated_log = tmp_path / "no_gen_log_test.jsonl"
    custom_logger = StructuredLogger(log_path=isolated_log)
    monkeypatch.setattr("src.api.structured_logger", custom_logger)

    # Simulation Situation A (0 chunk autorisé -> génération annulée)
    mock_response = RAGResponse(
        query="Question sans sources autorisées",
        answer="Aucun document accessible",
        user_id="bob",
        clearance_level=1,
        sources=[],
        retrieval_time_ms=14.0,
        generation_time_ms=0.0,
        total_time_ms=14.0,
        model_name="llama3.2",
        has_sufficient_context=False,
        chunks_retrieved_count=0
    )
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = mock_response
    monkeypatch.setattr("src.api.get_pipeline", lambda: mock_pipeline)

    client = TestClient(app)
    response = client.post(
        "/query",
        json={"query": "Question sans sources", "user_clearance": 1, "user_id": "bob"}
    )
    assert response.status_code == 200

    assert isolated_log.exists()
    lines = isolated_log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    event = json.loads(lines[0])
    # generation_time_ms doit être explicitement None / null
    assert event["generation_time_ms"] is None
    assert event["retrieval_time_ms"] == 14.0
    assert event["has_sufficient_context"] is False
    assert event["resolved_clearance_level"] == 1
