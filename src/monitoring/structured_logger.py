"""Journaliseur structuré au format JSON pour l'observabilité minimale de l'API RAG."""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any


DEFAULT_LOG_PATH = Path("logs/api_access.jsonl")


class StructuredLogger:
    """
    Gestionnaire de logs structurés au format JSON Lines (.jsonl).
    
    Garanties de sécurité et de conformité :
      - Enregistre exclusivement les métadonnées techniques et de performance.
      - Ne journalise JAMAIS les prompts complets ni le contenu des documents.
      - Ne journalise JAMAIS les réponses générées par le LLM.
      - Ne stocke AUCUNE donnée à caractère personnel ou confidentiel (RGPD / Secret d'affaires).
      - Écriture thread-safe via verrou d'exclusion mutuelle.
    """

    def __init__(self, log_path: Optional[Path] = None):
        env_path = os.getenv("RAG_LOG_PATH")
        if env_path:
            self.log_path = Path(env_path)
        else:
            self.log_path = log_path or DEFAULT_LOG_PATH

        self._lock = threading.Lock()
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Garantit l'existence du dossier de destination des journaux."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def log_event(self, event_data: Dict[str, Any]) -> None:
        """Écrit un événement structuré sous forme d'une ligne JSON (JSONL)."""
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        line = json.dumps(event_data, ensure_ascii=False)
        with self._lock:
            try:
                self._ensure_log_dir()
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                # La journalisation ne doit jamais faire échouer la requête principale
                pass

    def log_query_request(
        self,
        request_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        total_duration_ms: float,
        retrieval_time_ms: Optional[float] = None,
        generation_time_ms: Optional[float] = None,
        pipeline_time_ms: Optional[float] = None,
        has_sufficient_context: Optional[bool] = None,
        resolved_clearance_level: Optional[int] = None,
        error_type: Optional[str] = None
    ) -> None:
        """
        Journalise les métriques d'exécution complètes d'un appel RAG.
        Distinction stricte entre durée HTTP totale et latence du pipeline RAG.
        """
        event: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "total_duration_ms": round(float(total_duration_ms), 2),
            "retrieval_time_ms": round(float(retrieval_time_ms), 2) if retrieval_time_ms is not None else None,
            "generation_time_ms": round(float(generation_time_ms), 2) if generation_time_ms is not None else None,
            "pipeline_time_ms": round(float(pipeline_time_ms), 2) if pipeline_time_ms is not None else None,
            "has_sufficient_context": has_sufficient_context,
            "resolved_clearance_level": resolved_clearance_level,
            "error_type": error_type
        }
        self.log_event(event)

    def log_generic_request(
        self,
        request_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        total_duration_ms: float,
        error_type: Optional[str] = None
    ) -> None:
        """Journalise une requête HTTP générique (ex: /health, 404, erreurs de validation Pydantic)."""
        event: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "total_duration_ms": round(float(total_duration_ms), 2),
            "retrieval_time_ms": None,
            "generation_time_ms": None,
            "pipeline_time_ms": None,
            "has_sufficient_context": None,
            "resolved_clearance_level": None,
            "error_type": error_type
        }
        self.log_event(event)


# Instance globale singleton
_global_logger: Optional[StructuredLogger] = None


def get_structured_logger() -> StructuredLogger:
    """Retourne l'instance globale du journaliseur structuré."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger()
    return _global_logger

