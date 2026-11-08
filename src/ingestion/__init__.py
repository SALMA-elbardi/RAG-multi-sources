"""Package d'ingestion documentaire multi-sources de DataNova Technologies."""
import importlib.util
from pathlib import Path

from src.ingestion.models import NormalizedDocument, IngestionReport, MetadataValue
from src.ingestion.metadata_manager import MetadataManager, MetadataDiscrepancy
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.markdown_loader import MarkdownLoader
from src.ingestion.json_loader import JSONLoader
from src.ingestion.csv_loader import CSVLoader
from src.ingestion.document_validator import DocumentValidator
from src.ingestion.loader_factory import LoaderFactory, UnsupportedFormatError

# Coexistence avec src/ingestion.py racine
_legacy_path = Path(__file__).resolve().parent.parent / "ingestion.py"
CorpusIngestionPipeline = None
if _legacy_path.exists():
    try:
        _spec = importlib.util.spec_from_file_location("src._legacy_ingestion", _legacy_path)
        if _spec and _spec.loader:
            _legacy_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_legacy_mod)
            CorpusIngestionPipeline = getattr(_legacy_mod, "CorpusIngestionPipeline", None)
    except Exception:
        CorpusIngestionPipeline = None


def __getattr__(name: str):
    """Import paresseux de IngestionPipeline pour éviter le RuntimeWarning de runpy lors de l'exécution en CLI."""
    if name == "IngestionPipeline":
        from src.ingestion.ingestion_pipeline import IngestionPipeline
        return IngestionPipeline
    raise AttributeError(f"Le module '{__name__}' n'a pas d'attribut '{name}'")


__all__ = [
    "NormalizedDocument",
    "MetadataValue",
    "IngestionReport",
    "MetadataManager",
    "MetadataDiscrepancy",
    "PDFLoader",
    "MarkdownLoader",
    "JSONLoader",
    "CSVLoader",
    "DocumentValidator",
    "LoaderFactory",
    "UnsupportedFormatError",
    "IngestionPipeline",
    "CorpusIngestionPipeline",
]
