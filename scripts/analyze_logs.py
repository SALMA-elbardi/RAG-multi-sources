"""Script d'analyse des journaux d'observabilité de l'API RAG (Phase 9)."""
import sys
from pathlib import Path

# Inclusion de la racine du projet dans le PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.monitoring.metrics_analyzer import MetricsAnalyzer


def main():
    log_arg = sys.argv[1] if len(sys.argv) > 1 else None
    log_path = Path(log_arg) if log_arg else Path("logs/api_access.jsonl")

    analyzer = MetricsAnalyzer(log_path=log_path)
    print(analyzer.generate_report())


if __name__ == "__main__":
    main()

