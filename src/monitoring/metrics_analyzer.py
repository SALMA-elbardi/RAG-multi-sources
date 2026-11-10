"""Analyseur statistique de métriques et générateur de rapports d'observabilité pour le RAG."""
import json
import math
import os
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def calculate_percentile(values: List[float], percentile: float = 95.0) -> float:
    """
    Calcule un percentile (par exemple P95) selon la méthode mathématique du rang le plus proche (Nearest-Rank).
    
    Formule :
      1. Trier la liste en ordre croissant.
      2. Rang r = ceil((percentile / 100) * N)
      3. Index = max(0, min(r - 1, N - 1))
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return round(sorted_vals[0], 2)
    rank = math.ceil((percentile / 100.0) * n)
    index = max(0, min(rank - 1, n - 1))
    return round(sorted_vals[index], 2)


class MetricsAnalyzer:
    """
    Analyseur statistique de journaux JSONL pour le suivi opérationnel du système RAG.
    
    Indicateurs calculés :
      1. Nombre total de requêtes (Global, Succès, Échecs).
      2. Taux d'erreur HTTP.
      3. Latence moyenne et P95 de la requête HTTP totale.
      4. Temps moyen et P95 de retrieval documentaire (ChromaDB).
      5. Temps moyen et P95 de génération LLM locale (Ollama).
      6. Temps moyen et P95 du pipeline RAG interne.
      7. Disponibilité réelle du service Ollama (en direct via urllib.request standard).
      8. Contrôle de complétude documentaire (contexte suffisant vs insuffisant).
      9. Répartition par niveau d'accréditation RBAC.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = Path(log_path or os.getenv("RAG_LOG_PATH", "logs/api_access.jsonl"))

    def load_events(self) -> List[Dict[str, Any]]:
        """Charge et désérialise les événements JSON depuis le fichier journal."""
        events: List[Dict[str, Any]] = []
        if not self.log_path.exists():
            return events

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    record = json.loads(clean_line)
                    events.append(record)
                except json.JSONDecodeError:
                    # Ignorer les lignes corrompues sans interrompre l'analyse
                    continue
        return events

    def check_ollama_live(self, base_url: Optional[str] = None) -> Tuple[bool, str]:
        """
        Interroge le statut réel du serveur Ollama local (/api/version).
        Retourne (is_online, message_detail).
        Utilise exclusivement urllib.request (bibliothèque standard Python).
        """
        url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        endpoint = f"{url.rstrip('/')}/api/version"
        req = urllib.request.Request(
            endpoint,
            headers={"User-Agent": "DataNova-RAG-Monitor/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                status_code = response.getcode()
                if status_code == 200:
                    raw_data = response.read().decode("utf-8")
                    data = json.loads(raw_data)
                    version = data.get("version", "connue")
                    return True, f"En ligne (Ollama v{version} sur {url})"
                return False, f"Injoignable (Code HTTP {status_code} sur {url})"
        except Exception as e:
            return False, f"Hors ligne ({url}) : {str(e)}"

    def compute_metrics(self, events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Calcule l'ensemble des indicateurs statistiques prescrits par le cahier des charges."""
        if events is None:
            events = self.load_events()

        total_requests = len(events)
        if total_requests == 0:
            is_online, ollama_status = self.check_ollama_live()
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "error_rate_pct": 0.0,
                "mean_total_duration_ms": 0.0,
                "p95_total_duration_ms": 0.0,
                "mean_retrieval_time_ms": 0.0,
                "p95_retrieval_time_ms": 0.0,
                "mean_generation_time_ms": None,
                "p95_generation_time_ms": None,
                "generation_metrics_available": False,
                "generation_count": 0,
                "mean_pipeline_time_ms": 0.0,
                "p95_pipeline_time_ms": 0.0,
                "ollama_online": is_online,
                "ollama_status_detail": ollama_status,
                "sufficient_context_count": 0,
                "insufficient_context_count": 0,
                "requests_by_clearance": {},
                "requests_by_status": {},
                "error_types": {},
                "unavailable_metrics": [
                    "Consommation CPU/GPU temps réel (exclue pour respecter l'absence de dépendances lourdes)",
                    "Score de fidélité sémantique / Faithfulness (réservé aux benchmarks annotés hors production)"
                ]
            }

        # 1. Compteurs de statut HTTP
        successful = 0
        failed = 0
        status_counts: Dict[int, int] = {}
        error_types: Dict[str, int] = {}

        # 2. Séries temporelles de latence
        total_durations: List[float] = []
        retrieval_durations: List[float] = []
        generation_durations: List[float] = []
        pipeline_durations: List[float] = []

        # 3. Métriques RAG & RBAC
        sufficient_context = 0
        insufficient_context = 0
        clearance_counts: Dict[str, int] = {
            "Niveau 1 (Public)": 0,
            "Niveau 2 (Interne)": 0,
            "Niveau 3 (Restreint)": 0,
            "Non résolu / Non applicable": 0
        }

        for ev in events:
            sc = ev.get("status_code", 0)
            status_counts[sc] = status_counts.get(sc, 0) + 1

            if 200 <= sc < 400:
                successful += 1
            else:
                failed += 1

            err = ev.get("error_type")
            if err:
                error_types[err] = error_types.get(err, 0) + 1

            # Durée HTTP totale
            dur = ev.get("total_duration_ms")
            if dur is not None:
                total_durations.append(float(dur))

            # Durées RAG spécifiques
            ret = ev.get("retrieval_time_ms")
            if ret is not None:
                retrieval_durations.append(float(ret))

            # Exclusion stricte des valeurs None (générations non exécutées)
            gen = ev.get("generation_time_ms")
            if gen is not None:
                generation_durations.append(float(gen))

            pip = ev.get("pipeline_time_ms")
            if pip is not None:
                pipeline_durations.append(float(pip))

            # Complétude du contexte
            has_ctx = ev.get("has_sufficient_context")
            if has_ctx is True:
                sufficient_context += 1
            elif has_ctx is False:
                insufficient_context += 1

            # Niveaux RBAC
            clr = ev.get("resolved_clearance_level")
            if clr == 1:
                clearance_counts["Niveau 1 (Public)"] += 1
            elif clr == 2:
                clearance_counts["Niveau 2 (Interne)"] += 1
            elif clr == 3:
                clearance_counts["Niveau 3 (Restreint)"] += 1
            else:
                clearance_counts["Non résolu / Non applicable"] += 1

        error_rate = round((failed / total_requests) * 100.0, 2)
        mean_tot = round(sum(total_durations) / len(total_durations), 2) if total_durations else 0.0
        p95_tot = calculate_percentile(total_durations, 95.0)

        mean_ret = round(sum(retrieval_durations) / len(retrieval_durations), 2) if retrieval_durations else 0.0
        p95_ret = calculate_percentile(retrieval_durations, 95.0)

        # Calcul robuste de la latence de génération LLM
        if generation_durations:
            mean_gen: Optional[float] = round(sum(generation_durations) / len(generation_durations), 2)
            p95_gen: Optional[float] = calculate_percentile(generation_durations, 95.0)
            generation_metrics_available = True
        else:
            mean_gen = None
            p95_gen = None
            generation_metrics_available = False

        mean_pip = round(sum(pipeline_durations) / len(pipeline_durations), 2) if pipeline_durations else 0.0
        p95_pip = calculate_percentile(pipeline_durations, 95.0)

        is_online, ollama_status = self.check_ollama_live()

        return {
            "total_requests": total_requests,
            "successful_requests": successful,
            "failed_requests": failed,
            "error_rate_pct": error_rate,
            "mean_total_duration_ms": mean_tot,
            "p95_total_duration_ms": p95_tot,
            "mean_retrieval_time_ms": mean_ret,
            "p95_retrieval_time_ms": p95_ret,
            "mean_generation_time_ms": mean_gen,
            "p95_generation_time_ms": p95_gen,
            "generation_metrics_available": generation_metrics_available,
            "generation_count": len(generation_durations),
            "mean_pipeline_time_ms": mean_pip,
            "p95_pipeline_time_ms": p95_pip,
            "ollama_online": is_online,
            "ollama_status_detail": ollama_status,
            "sufficient_context_count": sufficient_context,
            "insufficient_context_count": insufficient_context,
            "requests_by_clearance": clearance_counts,
            "requests_by_status": status_counts,
            "error_types": error_types,
            "unavailable_metrics": [
                "Consommation CPU/GPU temps réel (exclue pour respecter l'absence de dépendances lourdes)",
                "Score de fidélité sémantique / Faithfulness (réservé aux benchmarks annotés hors production)"
            ]
        }

    def generate_report(self) -> str:
        """Génère un rapport textuel synthétique, clair et professionnel."""
        m = self.compute_metrics()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = [
            "=" * 80,
            "DATANOVA TECHNOLOGIES — RAPPORT D'OBSERVABILITÉ DU RAG (PHASE 9)",
            "=" * 80,
            f"Fichier de journalisation : {self.log_path}",
            f"Date d'analyse            : {now_str}",
            "",
            "[1] DISPONIBILITÉ DU SERVICE & VOLUMÉTRIE",
            f"  - Serveur LLM local (Ollama)  : {'EN LIGNE' if m['ollama_online'] else 'HORS LIGNE'} — {m['ollama_status_detail']}",
            f"  - Nombre total de requêtes    : {m['total_requests']}",
            f"  - Requêtes réussies (2xx/3xx) : {m['successful_requests']}",
            f"  - Requêtes en échec (4xx/5xx)  : {m['failed_requests']}",
            f"  - Taux d'erreur HTTP global   : {m['error_rate_pct']} %",
            "",
            "[2] TEMPS DE TRAITEMENT ET LATENCES (Moyenne vs P95)",
            "  * Durée HTTP globale (réseau + middleware + logique) :",
            f"      Moyenne : {m['mean_total_duration_ms']} ms | P95 : {m['p95_total_duration_ms']} ms",
            "  * Temps de recherche documentaire (Retrieval ChromaDB) :",
            f"      Moyenne : {m['mean_retrieval_time_ms']} ms | P95 : {m['p95_retrieval_time_ms']} ms",
            "  * Temps de génération locale (Inférence Ollama Llama 3.2) :"
        ]

        if m.get("generation_metrics_available") and m["mean_generation_time_ms"] is not None:
            report.append(
                f"      Moyenne : {m['mean_generation_time_ms']} ms | P95 : {m['p95_generation_time_ms']} ms ({m['generation_count']} génération(s) effective(s))"
            )
        else:
            report.append(
                "      Indisponible (aucune génération LLM effective enregistrée)"
            )

        report.extend([
            "  * Temps de traitement interne du pipeline RAG :",
            f"      Moyenne : {m['mean_pipeline_time_ms']} ms | P95 : {m['p95_pipeline_time_ms']} ms",
            "",
            "[3] QUALITÉ ET CONTRÔLE CONTEXTUEL (RAG)",
            f"  - Réponses avec contexte suffisant   : {m['sufficient_context_count']}",
            f"  - Réponses avec contexte insuffisant : {m['insufficient_context_count']}",
            "    (Remarque : le contrôle contextuel détecte les manques formulés par le LLM mais ne",
            "     se substitue pas à l'évaluation formelle de fidélité sur jeu annoté).",
            "",
            "[4] RÉPARTITION DES ACCRÉDITATIONS RBAC RÉSOLUES CÔTÉ SERVEUR"
        ])

        for level_name, count in m["requests_by_clearance"].items():
            report.append(f"  - {level_name:<28} : {count} requête(s)")

        if m["requests_by_status"]:
            report.append("")
            report.append("[5] RÉPARTITION DES CODES DE STATUT HTTP")
            for code, count in sorted(m["requests_by_status"].items()):
                report.append(f"  - HTTP {code} : {count} appel(s)")

        if m["error_types"]:
            report.append("")
            report.append("[6] TYPES D'ERREURS RENCONTRÉES")
            for err_type, count in sorted(m["error_types"].items()):
                report.append(f"  - {err_type} : {count} occurrence(s)")

        report.append("")
        report.append("[7] INDICATEURS NON INSTRUMENTÉS (JUSTIFICATION CONFORME AU CAHIER DES CHARGES)")
        for unavail in m["unavailable_metrics"]:
            report.append(f"  - [INDISPONIBLE] {unavail}")

        report.append("=" * 80)
        return "\n".join(report)

        for level_name, count in m["requests_by_clearance"].items():
            report.append(f"  - {level_name:<28} : {count} requête(s)")

        if m["requests_by_status"]:
            report.append("")
            report.append("[5] RÉPARTITION DES CODES DE STATUT HTTP")
            for code, count in sorted(m["requests_by_status"].items()):
                report.append(f"  - HTTP {code} : {count} appel(s)")

        if m["error_types"]:
            report.append("")
            report.append("[6] TYPES D'ERREURS RENCONTRÉES")
            for err_type, count in sorted(m["error_types"].items()):
                report.append(f"  - {err_type} : {count} occurrence(s)")

        report.append("")
        report.append("[7] INDICATEURS NON INSTRUMENTÉS (JUSTIFICATION CONFORME AU CAHIER DES CHARGES)")
        for unavail in m["unavailable_metrics"]:
            report.append(f"  - [INDISPONIBLE] {unavail}")

        report.append("=" * 80)
        return "\n".join(report)


if __name__ == "__main__":
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else None
    analyzer = MetricsAnalyzer(log_path=Path(log_file) if log_file else None)
    print(analyzer.generate_report())
