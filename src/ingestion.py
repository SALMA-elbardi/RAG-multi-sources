"""Module d'ingestion multi-sources (PDF, Markdown, FAQ) avec traçabilité stricte."""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.schema import DocumentChunk, ChunkMetadata, SensitivityLevel


class CorpusIngestionPipeline:
    """
    Pipeline d'ingestion normalisé garantissant que chaque chunk
    conserve strictement ses métadonnées de provenance et son niveau de sensibilité.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @staticmethod
    def generate_chunk_id(source: str, index: int, content: str) -> str:
        """Génère un identifiant unique et déterministe pour chaque chunk."""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        clean_source = Path(source).stem.replace(" ", "_").lower()
        return f"{clean_source}#c{index}_{content_hash}"

    def ingest_pdf(
        self,
        file_path: Path,
        default_sensitivity: int = SensitivityLevel.INTERNAL,
        date_version: str = "2026-11-08"
    ) -> List[DocumentChunk]:
        """Extrait et découpe le contenu d'un fichier PDF avec provenance."""
        reader = PdfReader(str(file_path))
        full_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                full_text.append(f"--- Page {page_num} ---\n{text}")

        combined_text = "\n\n".join(full_text)
        if not combined_text.strip():
            return []

        raw_chunks = self.text_splitter.split_text(combined_text)
        chunks: List[DocumentChunk] = []
        for idx, text_block in enumerate(raw_chunks):
            chunk_id = self.generate_chunk_id(file_path.name, idx, text_block)
            metadata = ChunkMetadata(
                id=chunk_id,
                source=file_path.name,
                date_version=date_version,
                doc_type="pdf",
                sensitivity_level=default_sensitivity
            )
            chunks.append(DocumentChunk(content=text_block, metadata=metadata))
        return chunks

    def ingest_markdown(
        self,
        file_path: Path,
        default_sensitivity: int = SensitivityLevel.PUBLIC,
        date_version: str = "2026-11-08"
    ) -> List[DocumentChunk]:
        """Extrait et découpe le contenu d'un fichier Markdown avec provenance."""
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            return []

        raw_chunks = self.text_splitter.split_text(content)
        chunks: List[DocumentChunk] = []
        for idx, text_block in enumerate(raw_chunks):
            chunk_id = self.generate_chunk_id(file_path.name, idx, text_block)
            metadata = ChunkMetadata(
                id=chunk_id,
                source=file_path.name,
                date_version=date_version,
                doc_type="markdown",
                sensitivity_level=default_sensitivity
            )
            chunks.append(DocumentChunk(content=text_block, metadata=metadata))
        return chunks

    def ingest_faq_json(self, file_path: Path) -> List[DocumentChunk]:
        """
        Ingère une base de questions/réponses au format JSON.
        Chaque entrée peut définir son propre niveau de sensibilité et sa version.
        Format attendu : list of dicts {question, answer, sensitivity_level, date_version, id?}
        """
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Le fichier FAQ JSON doit contenir une liste d'objets Q/A.")

        chunks: List[DocumentChunk] = []
        for idx, entry in enumerate(data):
            question = entry.get("question", "").strip()
            answer = entry.get("answer", "").strip()
            if not question or not answer:
                continue

            content = f"Question : {question}\nRéponse : {answer}"
            sensitivity = int(entry.get("sensitivity_level", SensitivityLevel.PUBLIC))
            date_ver = str(entry.get("date_version", "2026-11-08"))
            custom_id = entry.get("id") or self.generate_chunk_id(file_path.name, idx, content)

            metadata = ChunkMetadata(
                id=custom_id,
                source=file_path.name,
                date_version=date_ver,
                doc_type="faq",
                sensitivity_level=sensitivity
            )
            chunks.append(DocumentChunk(content=content, metadata=metadata))
        return chunks

    def ingest_directory(self, directory_path: Path) -> List[DocumentChunk]:
        """Parcourt récursivement un dossier et ingère tous les documents multi-sources trouvés."""
        all_chunks: List[DocumentChunk] = []
        if not directory_path.exists():
            return all_chunks

        for path in directory_path.rglob("*"):
            if path.is_file():
                ext = path.suffix.lower()
                if ext == ".pdf":
                    all_chunks.extend(self.ingest_pdf(path))
                elif ext in [".md", ".markdown"]:
                    all_chunks.extend(self.ingest_markdown(path))
                elif ext == ".json" and "faq" in path.name.lower():
                    all_chunks.extend(self.ingest_faq_json(path))

        return all_chunks

