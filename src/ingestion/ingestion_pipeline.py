"""Orchestrateur du pipeline d'ingestion documentaire multi-sources de DataNova Technologies."""
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.ingestion.models import NormalizedDocument, IngestionReport
from src.ingestion.metadata_manager import MetadataManager
from src.ingestion.loader_factory import LoaderFactory, UnsupportedFormatError
from src.ingestion.document_validator import DocumentValidator
from src.ingestion.pdf_loader import PDFLoader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrateur central du pipeline d'ingestion documentaire multi-sources.
    Règles strictes :
      1. Scanne exclusivement 'data/raw/'.
      2. Exclut formellement 'inventory.json' de l'indexation (utilisé comme référentiel).
      3. Interdit tout accès ou chargement de 'data/test_fixtures/'.
      4. Traite chaque format via LoaderFactory et gère les signatures de retour hétérogènes.
      5. Soumet tous les documents au DocumentValidator.
      6. Résilience : l'échec sur un fichier isolé n'interrompt pas le reste du traitement.
      7. Calcule dynamiquement les statistiques réelles (aucun chiffre codé en dur).
    """

    FORBIDDEN_PATH_SEGMENTS = {"test_fixtures"}

    def __init__(
        self,
        raw_data_dir: Optional[Path] = None,
        inventory_path: Optional[Path] = None
    ):
        self.raw_data_dir = raw_data_dir or Path("data/raw")
        self.inventory_path = inventory_path or Path("data/raw/inventory.json")
        self.metadata_manager = MetadataManager(self.inventory_path)
        self.loader_factory = LoaderFactory(self.metadata_manager)
        self.validator = DocumentValidator()

    def _is_path_allowed(self, path: Path) -> bool:
        """Vérifie que le chemin appartient strictement à data/raw et n'est pas une fixture de test."""
        posix_str = path.as_posix().lower()

        # Interdiction absolue de tout fichier situé dans le dossier test_fixtures
        if "test_fixtures" in posix_str:
            logger.warning(f"Chemin de test fixture exclu de l'ingestion : {path}")
            return False

        # Doit appartenir au répertoire raw_data_dir
        try:
            path.resolve().relative_to(self.raw_data_dir.resolve())
            return True
        except ValueError:
            return False

    def scan_files(self) -> List[Path]:
        """
        Détecte récursivement les fichiers du corpus actif dans data/raw/.
        Exclut inventory.json et les formats non supportés.
        """
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Répertoire du corpus actif introuvable : {self.raw_data_dir}")

        detected_files: List[Path] = []
        for file_path in sorted(self.raw_data_dir.rglob("*")):
            if not file_path.is_file():
                continue

            # Vérification de sécurité du chemin
            if not self._is_path_allowed(file_path):
                continue

            # inventory.json est le référentiel de métadonnées, pas un document à indexer
            if file_path.name.lower() == "inventory.json":
                continue

            # Vérification du format supporté
            if self.loader_factory.is_supported(file_path):
                detected_files.append(file_path)
            else:
                logger.info(f"Fichier ignoré (extension non prise en charge) : {file_path.name}")

        return detected_files

    def run(self) -> Tuple[List[NormalizedDocument], IngestionReport]:
        """
        Exécute l'ingestion complète du corpus actif et retourne (documents_valides, rapport).
        """
        start_time = time.time()
        report = IngestionReport()

        files_to_process = self.scan_files()
        report.total_files_detected = len(files_to_process)

        all_valid_documents: List[NormalizedDocument] = []
        raw_documents_count = 0

        for file_path in files_to_process:
            ext = file_path.suffix.lower()
            format_key = ext.lstrip(".")
            if format_key in ("md", "markdown"):
                format_key = "markdown"

            report.files_by_format[format_key] = report.files_by_format.get(format_key, 0) + 1

            try:
                loader = self.loader_factory.get_loader(file_path)

                # Gestion compatible de la signature spécifique de PDFLoader (docs, ignored_pages)
                if isinstance(loader, PDFLoader):
                    loaded_docs, ignored_pages = loader.load(file_path)
                    if ignored_pages:
                        report.ignored_pages_count += len(ignored_pages)
                        report.ignored_pages_details.extend(ignored_pages)
                else:
                    loaded_docs = loader.load(file_path)

                raw_documents_count += len(loaded_docs)

                # Validation stricte du lot de documents
                valid_docs, rejected_details, warnings = self.validator.validate_batch(loaded_docs)

                # Comptabilisation des rejets
                if rejected_details:
                    report.rejected_documents_count += len(rejected_details)
                    for rej in rejected_details:
                        report.errors.append(f"Rejet [{rej['id']}] ({rej['source']}) : {rej['reason']}")

                # Comptabilisation des avertissements
                if warnings:
                    for warn in warnings:
                        report.anomalies.append(f"Avertissement [{warn['id']}] : {warn['warning']}")

                # Agrégation des documents valides
                for doc in valid_docs:
                    all_valid_documents.append(doc)
                    doc_type = doc.doc_type
                    report.documents_by_format[doc_type] = report.documents_by_format.get(doc_type, 0) + 1
                    sens = doc.sensitivity_level
                    report.documents_by_sensitivity[sens] = report.documents_by_sensitivity.get(sens, 0) + 1

            except Exception as e:
                # Résilience MLOps : l'erreur est consignée sans interrompre les autres fichiers
                logger.error(f"Erreur d'ingestion sur '{file_path}' : {e}")
                report.errors_count += 1
                report.errors.append(f"Erreur fichier '{file_path.name}' : {str(e)}")

        # Collecte des divergences d'inventaire détectées
        for disc in self.metadata_manager.discrepancies:
            report.anomalies.append(f"Divergence inventaire [{disc.document_id}] : {disc.description}")

        report.total_documents_loaded = len(all_valid_documents)
        report.execution_time_seconds = time.time() - start_time

        return all_valid_documents, report

    @staticmethod
    def print_report(report: IngestionReport) -> None:
        """Affiche un tableau clair et structuré du rapport d'ingestion dans la console."""
        print("\n" + "=" * 80)
        print("RAPPORT D'INGESTION DOCUMENTAIRE DU CORPUS — DATANOVA TECHNOLOGIES")
        print("=" * 80)
        print(f"Fichiers détectés dans data/raw/     : {report.total_files_detected}")
        for fmt, count in sorted(report.files_by_format.items()):
            print(f"  • Format {fmt.upper():<10}             : {count} fichier(s)")

        print("-" * 80)
        print(f"Documents normalisés validés en mémoire : {report.total_documents_loaded}")
        for doc_type, count in sorted(report.documents_by_format.items()):
            print(f"  • Type {doc_type.upper():<12}           : {count} document(s)")

        print("-" * 80)
        print("Répartition par niveau de sensibilité RBAC :")
        for sens_lvl in (1, 2, 3):
            label = {1: "Niveau 1 (Public / Interne)", 2: "Niveau 2 (Restreint)", 3: "Niveau 3 (Confidentiel)"}.get(sens_lvl)
            count = report.documents_by_sensitivity.get(sens_lvl, 0)
            print(f"  • {label:<35} : {count} document(s)")

        print("-" * 80)
        print(f"Qualité et anomalies :")
        print(f"  • Documents rejetés                  : {report.rejected_documents_count}")
        print(f"  • Pages PDF ignorées (blanches)      : {report.ignored_pages_count}")
        print(f"  • Erreurs de traitement              : {report.errors_count}")
        print(f"  • Avertissements / Divergences       : {len(report.anomalies)}")

        if report.ignored_pages_details:
            print("\nDétail des pages ignorées :")
            for ign in report.ignored_pages_details:
                print(f"  - {ign['file']} (page {ign['page_number']}) : {ign['reason']}")

        if report.errors:
            print("\nDétail des erreurs :")
            for err in report.errors:
                print(f"  - {err}")

        if report.anomalies:
            print("\nDétail des avertissements et divergences :")
            for a in report.anomalies:
                print(f"  - {a}")

        print("-" * 80)
        print(f"Temps d'exécution total               : {report.execution_time_seconds:.3f} seconde(s)")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = IngestionPipeline()
    docs, report = pipeline.run()
    IngestionPipeline.print_report(report)

