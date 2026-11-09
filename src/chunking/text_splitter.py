"""Splitters modulaires pour le découpage documentaire (stratégies fixe et structurelle)."""
import re
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.models import NormalizedDocument
from src.chunking.models import ChunkDocument
from src.chunking.chunking_config import ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)


def is_syntax_noise(text: str) -> bool:
    """
    Détecte les fragments purement syntaxiques sans aucune valeur sémantique
    (ex: lignes de séparation horizontales '---', '***', '___' isolées).
    """
    cleaned = text.strip()
    if not cleaned:
        return True
    if re.fullmatch(r"[-*_\s=]{2,}", cleaned):
        return True
    return False


def generate_chunk_id(parent_doc_id: str, strategy: str, chunk_index: int, content: str) -> str:
    """
    Génère un identifiant déterministe, stable et unique pour chaque chunk.
    Combine :
      - Identifiant du document parent normalisé (parent_doc_id, ex: 'DOC-SEC-001#p1', 'FAQ-DE-001')
      - Préfixe de la stratégie ('fix' ou 'str')
      - Index séquentiel du chunk (0, 1, 2...)
      - Hash SHA-256 du contenu textuel réel du chunk (12 caractères hexadécimaux)
      
    Garanties :
      - Aucun timestamp (reproductibilité binaire parfaite entre runs successifs).
      - Déterminisme absolu : même entrée + même config = même identifiant.
      - Zéro collision entre documents ou entre stratégies.
    """
    clean_parent = str(parent_doc_id).strip().replace(" ", "_")
    content_hash = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:12]
    strat_tag = "fix" if str(strategy).lower().startswith("fix") else "str"
    return f"{clean_parent}_{strat_tag}_c{chunk_index}_{content_hash}"


class BaseChunkSplitter(ABC):
    """Classe de base abstraite pour les découpeurs de documents."""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def split_document(self, doc: NormalizedDocument) -> List[ChunkDocument]:
        """Découpe un NormalizedDocument unitaire en une liste de ChunkDocument."""
        pass

    def split_documents(self, docs: List[NormalizedDocument]) -> List[ChunkDocument]:
        """Découpe un lot complet de documents normalisés."""
        all_chunks: List[ChunkDocument] = []
        for doc in docs:
            doc_chunks = self.split_document(doc)
            all_chunks.extend(doc_chunks)
        return all_chunks


class FixedTextSplitter(BaseChunkSplitter):
    """
    Stratégie A — Chunking fixe uniforme.
    Applique un découpage régulier par taille indicative et chevauchement
    via RecursiveCharacterTextSplitter sur tous les formats documentaires.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators
        )

    def split_document(self, doc: NormalizedDocument) -> List[ChunkDocument]:
        clean_text = doc.page_content.strip() if doc.page_content else ""
        if not clean_text:
            return []

        raw_chunks = self.text_splitter.split_text(clean_text)
        if not raw_chunks:
            return []

        # Filtrage du bruit syntaxique pur (ex: lignes horizontales '---' isolées)
        filtered_chunks = [tb.strip() for tb in raw_chunks if tb.strip() and not is_syntax_noise(tb)]
        if not filtered_chunks:
            return []

        total_chunks = len(filtered_chunks)
        chunks: List[ChunkDocument] = []

        for idx, text_block in enumerate(filtered_chunks):
            chunk_id = generate_chunk_id(doc.id, "fix", idx, text_block)
            
            # Héritage strict des métadonnées du document parent
            chunk_metadata: Dict[str, Any] = dict(doc.metadata)
            
            # Injection des métadonnées obligatoires de chunking (sans timestamp)
            chunk_metadata["chunk_id"] = chunk_id
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["total_chunks"] = total_chunks
            chunk_metadata["chunking_strategy"] = ChunkingStrategy.FIXED.value
            chunk_metadata["parent_doc_id"] = doc.id

            chunks.append(
                ChunkDocument(
                    chunk_id=chunk_id,
                    content=text_block,
                    metadata=chunk_metadata
                )
            )

        return chunks


class StructuralTextSplitter(BaseChunkSplitter):
    """
    Stratégie B — Chunking structurel adapté à la sémantique de chaque source :
      - FAQ : Maintien de la paire Question / Réponse dans un chunk unitaire.
      - CSV : Maintien de la fiche dataset complète (12 colonnes) dans un chunk unitaire.
      - Markdown : Préservation des sections cohérentes selon les titres (# et ##).
                   Le sous-découpage récursif intervient UNIQUEMENT si la section est trop volumineuse.
      - PDF : Respect strict des frontières de page et découpage intra-page par paragraphes.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.md_sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _split_markdown_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Découpe un document Markdown en sections cohérentes basées sur les titres (# et ##).
        Nettoie les lignes de séparation '---' en fin de section pour éviter l'émission
        de fragments de bruit résiduels.
        """
        lines = text.splitlines(keepends=True)
        sections: List[Tuple[str, str]] = []
        current_heading = "Préambule"
        current_lines: List[str] = []

        for line in lines:
            header_match = re.match(r"^(#{1,2})\s+(.+)$", line.strip())
            if header_match:
                if current_lines:
                    section_body = "".join(current_lines).strip()
                    # Nettoie les séparateurs horizontaux '---' résiduels en fin de section
                    section_body = re.sub(r"\n+[-*_\s=]{2,}\s*$", "", section_body).strip()
                    if section_body and not is_syntax_noise(section_body):
                        sections.append((current_heading, section_body))
                    current_lines = []
                current_heading = header_match.group(2).strip()
            current_lines.append(line)

        if current_lines:
            section_body = "".join(current_lines).strip()
            section_body = re.sub(r"\n+[-*_\s=]{2,}\s*$", "", section_body).strip()
            if section_body and not is_syntax_noise(section_body):
                sections.append((current_heading, section_body))

        return sections

    def _process_markdown(self, doc: NormalizedDocument) -> List[ChunkDocument]:
        """Découpage structurel d'un document Markdown préservant les sections cohérentes."""
        clean_text = doc.page_content.strip()
        if not clean_text:
            return []

        sections = self._split_markdown_sections(clean_text)
        if not sections:
            sections = [("Document Complet", clean_text)]

        raw_chunks: List[Tuple[str, str]] = []
        for heading, section_content in sections:
            # Si la section respecte la taille maximale de section, elle reste intacte
            if len(section_content) <= self.config.markdown_max_section_size:
                if not is_syntax_noise(section_content):
                    raw_chunks.append((section_content, heading))
            else:
                # Sous-découpage récursif UNIQUEMENT pour les sections dépassant le seuil
                sub_blocks = self.md_sub_splitter.split_text(section_content)
                for sub in sub_blocks:
                    clean_sub = sub.strip()
                    if clean_sub and not is_syntax_noise(clean_sub):
                        raw_chunks.append((clean_sub, heading))

        total_chunks = len(raw_chunks)
        chunks: List[ChunkDocument] = []

        for idx, (text_block, section_title) in enumerate(raw_chunks):
            chunk_id = generate_chunk_id(doc.id, "str", idx, text_block)
            meta: Dict[str, Any] = dict(doc.metadata)
            meta["chunk_id"] = chunk_id
            meta["chunk_index"] = idx
            meta["total_chunks"] = total_chunks
            meta["chunking_strategy"] = ChunkingStrategy.STRUCTURAL.value
            meta["section"] = section_title
            meta["parent_doc_id"] = doc.id

            chunks.append(
                ChunkDocument(
                    chunk_id=chunk_id,
                    content=text_block,
                    metadata=meta
                )
            )

        return chunks

    def _process_faq(self, doc: NormalizedDocument) -> List[ChunkDocument]:
        """Préservation de l'intégrité de la paire Question / Réponse."""
        clean_text = doc.page_content.strip()
        if not clean_text or is_syntax_noise(clean_text):
            return []

        if self.config.preserve_faq_atomicity and len(clean_text) <= 1500:
            blocks = [clean_text]
        else:
            blocks = self.pdf_splitter.split_text(clean_text)

        filtered_blocks = [b.strip() for b in blocks if b.strip() and not is_syntax_noise(b)]
        total_chunks = len(filtered_blocks)
        chunks: List[ChunkDocument] = []

        for idx, text_block in enumerate(filtered_blocks):
            chunk_id = generate_chunk_id(doc.id, "str", idx, text_block)
            meta: Dict[str, Any] = dict(doc.metadata)
            meta["chunk_id"] = chunk_id
            meta["chunk_index"] = idx
            meta["total_chunks"] = total_chunks
            meta["chunking_strategy"] = ChunkingStrategy.STRUCTURAL.value
            meta["parent_doc_id"] = doc.id

            chunks.append(
                ChunkDocument(
                    chunk_id=chunk_id,
                    content=text_block,
                    metadata=meta
                )
            )

        return chunks

    def _process_csv(self, doc: NormalizedDocument) -> List[ChunkDocument]:
        """Préservation de l'intégrité de la fiche dataset du catalogue CSV."""
        clean_text = doc.page_content.strip()
        if not clean_text or is_syntax_noise(clean_text):
            return []

        if self.config.preserve_csv_atomicity and len(clean_text) <= 1500:
            blocks = [clean_text]
        else:
            blocks = self.pdf_splitter.split_text(clean_text)

        filtered_blocks = [b.strip() for b in blocks if b.strip() and not is_syntax_noise(b)]
        total_chunks = len(filtered_blocks)
        chunks: List[ChunkDocument] = []

        for idx, text_block in enumerate(filtered_blocks):
            chunk_id = generate_chunk_id(doc.id, "str", idx, text_block)
            meta: Dict[str, Any] = dict(doc.metadata)
            meta["chunk_id"] = chunk_id
            meta["chunk_index"] = idx
            meta["total_chunks"] = total_chunks
            meta["chunking_strategy"] = ChunkingStrategy.STRUCTURAL.value
            meta["parent_doc_id"] = doc.id

            chunks.append(
                ChunkDocument(
                    chunk_id=chunk_id,
                    content=text_block,
                    metadata=meta
                )
            )

        return chunks

    def _process_pdf(self, doc: NormalizedDocument) -> List[ChunkDocument]:
        """Découpage intra-page du PDF respectant rigoureusement les frontières de page."""
        clean_text = doc.page_content.strip()
        if not clean_text or is_syntax_noise(clean_text):
            return []

        blocks = self.pdf_splitter.split_text(clean_text)
        filtered_blocks = [b.strip() for b in blocks if b.strip() and not is_syntax_noise(b)]
        if not filtered_blocks:
            return []

        total_chunks = len(filtered_blocks)
        chunks: List[ChunkDocument] = []

        for idx, text_block in enumerate(filtered_blocks):
            chunk_id = generate_chunk_id(doc.id, "str", idx, text_block)
            meta: Dict[str, Any] = dict(doc.metadata)
            meta["chunk_id"] = chunk_id
            meta["chunk_index"] = idx
            meta["total_chunks"] = total_chunks
            meta["chunking_strategy"] = ChunkingStrategy.STRUCTURAL.value
            meta["parent_doc_id"] = doc.id

            chunks.append(
                ChunkDocument(
                    chunk_id=chunk_id,
                    content=text_block,
                    metadata=meta
                )
            )

        return chunks

    def split_document(self, doc: NormalizedDocument) -> List[ChunkDocument]:
        """Achemine le document vers la méthode de découpage structurel appropriée."""
        doc_type = str(doc.doc_type).lower()

        if doc_type == "faq":
            return self._process_faq(doc)
        elif doc_type == "csv":
            return self._process_csv(doc)
        elif doc_type == "markdown":
            return self._process_markdown(doc)
        elif doc_type == "pdf":
            return self._process_pdf(doc)
        else:
            clean_text = doc.page_content.strip()
            if not clean_text or is_syntax_noise(clean_text):
                return []
            blocks = self.pdf_splitter.split_text(clean_text)
            filtered = [b.strip() for b in blocks if b.strip() and not is_syntax_noise(b)]
            total = len(filtered)
            chunks: List[ChunkDocument] = []
            for idx, text_block in enumerate(filtered):
                chunk_id = generate_chunk_id(doc.id, "str", idx, text_block)
                meta = dict(doc.metadata)
                meta["chunk_id"] = chunk_id
                meta["chunk_index"] = idx
                meta["total_chunks"] = total
                meta["chunking_strategy"] = ChunkingStrategy.STRUCTURAL.value
                meta["parent_doc_id"] = doc.id
                chunks.append(ChunkDocument(chunk_id=chunk_id, content=text_block, metadata=meta))
            return chunks
