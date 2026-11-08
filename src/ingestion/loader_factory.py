"""Factory pour l'instanciation et la réutilisation des loaders selon le format de fichier."""
from pathlib import Path
from typing import Optional, Union

from src.ingestion.metadata_manager import MetadataManager
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.markdown_loader import MarkdownLoader
from src.ingestion.json_loader import JSONLoader
from src.ingestion.csv_loader import CSVLoader

AnyLoader = Union[PDFLoader, MarkdownLoader, JSONLoader, CSVLoader]


class UnsupportedFormatError(ValueError):
    """Exception levée lorsqu'une extension de fichier n'est pas prise en charge."""
    pass


class LoaderFactory:
    """
    Fabrique centrale associant chaque format de fichier à son loader dédié.
    Maintient un cache interne des loaders pour éviter les réinstanciations inutiles
    et partage la même instance de MetadataManager.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".json", ".csv"}

    def __init__(self, metadata_manager: MetadataManager):
        self.metadata_manager = metadata_manager
        # Cache interne des instances de loaders
        self._pdf_loader = PDFLoader(metadata_manager)
        self._md_loader = MarkdownLoader(metadata_manager)
        self._json_loader = JSONLoader(metadata_manager)
        self._csv_loader = CSVLoader(metadata_manager)

    def get_loader(self, file_path: Path) -> AnyLoader:
        """
        Retourne l'instance du loader approprié pour le fichier donné.
        Lève UnsupportedFormatError si l'extension n'est pas reconnue.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._pdf_loader
        elif suffix in (".md", ".markdown"):
            return self._md_loader
        elif suffix == ".json":
            return self._json_loader
        elif suffix == ".csv":
            return self._csv_loader
        else:
            raise UnsupportedFormatError(
                f"Extension '{suffix}' non prise en charge pour le fichier '{file_path.name}'. "
                f"Formats supportés : {sorted(list(self.SUPPORTED_EXTENSIONS))}"
            )

    def is_supported(self, file_path: Path) -> bool:
        """Indique si l'extension du fichier fait partie des formats supportés."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
