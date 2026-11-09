"""Orchestrateur du pipeline d'indexation vectorielle ChromaDB pour DataNova Technologies."""
import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.chunking.models import ChunkDocument
from src.embeddings.embedding_config import EmbeddingConfig
from src.embeddings.tokenizer_inspector import TokenizerInspector, TokenAuditReport
from src.embeddings.embedding_manager import EmbeddingManager
from src.embeddings.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


@dataclass
class IndexingReport:
    """Rapport statistique et qualitatif d'exécution de l'indexation ChromaDB."""
    chunk_file_source: str
    chunking_strategy: str
    model_name: str
    model_revision: str
    embedding_dim: int
    batch_size: int
    collection_name: str
    persist_directory: str
    total_chunks_loaded: int = 0
    total_chunks_indexed: int = 0
    mode: str = "rebuild"
    execution_time_seconds: float = 0.0
    status: str = "success"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_file_source": self.chunk_file_source,
            "chunking_strategy": self.chunking_strategy,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "total_chunks_loaded": self.total_chunks_loaded,
            "total_chunks_indexed": self.total_chunks_indexed,
            "mode": self.mode,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "status": self.status,
            "error_message": self.error_message
        }


class IndexationPipeline:
    """
    Pipeline d'indexation vectorielle orchestrant :
      1. Chargement des chunks JSON de la Phase 5.
      2. Contrôle pré-inférence de la longueur tokenisée (tolérance zéro pour la troncature silencieuse).
      3. Calcul des embeddings via intfloat/multilingual-e5-small (préfixe 'passage: ' en mémoire).
      4. Insertion persistante dans ChromaDB (mode rebuild ou upsert).
      5. Métrologie exacte du temps d'exécution réel et rapport complet.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        tokenizer_inspector: Optional[TokenizerInspector] = None,
        chroma_store: Optional[ChromaStore] = None
    ):
        self.config = config or EmbeddingConfig.create_structural_config()
        self.embedding_manager = embedding_manager or EmbeddingManager(self.config)
        self.tokenizer_inspector = tokenizer_inspector or TokenizerInspector(self.config)
        self.chroma_store = chroma_store or ChromaStore(self.config)

    def load_chunks_from_json(self, file_path: Path) -> List[ChunkDocument]:
        """Charge et valide les chunks sérialisés issus de la Phase 5."""
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier de chunks introuvable : {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_chunks = data.get("chunks", [])
        if not raw_chunks:
            raise ValueError(f"Le fichier '{file_path}' ne contient aucun chunk sous la clé 'chunks'.")

        chunks = [ChunkDocument(**item) for item in raw_chunks]
        logger.info(f"{len(chunks)} chunks chargés depuis '{file_path}'.")
        return chunks

    def run(self, mode: str = "rebuild") -> Tuple[IndexingReport, TokenAuditReport]:
        """
        Exécute le pipeline d'indexation complet.
        Modes disponibles : 'rebuild' (Blue/Green staging sécurisé) ou 'upsert' (incrémental).
        """
        start_time = time.perf_counter()

        report = IndexingReport(
            chunk_file_source=str(self.config.chunk_file_path),
            chunking_strategy=self.config.chunking_strategy,
            model_name=self.config.model_name,
            model_revision=self.config.model_revision,
            embedding_dim=self.config.embedding_dim,
            batch_size=self.config.batch_size,
            collection_name=self.config.collection_name,
            persist_directory=str(self.config.persist_directory),
            mode=mode
        )

        try:
            # 1. Chargement des chunks
            chunks = self.load_chunks_from_json(self.config.chunk_file_path)
            report.total_chunks_loaded = len(chunks)

            # 2. Audit de la longueur tokenisée (Anti-troncature silencieuse)
            token_report = self.tokenizer_inspector.inspect_chunks(chunks)
            if token_report.has_overflow:
                overflow_details = token_report.overflow_chunks_details
                err_msg = (
                    f"Dépassement de token détecté sur {len(overflow_details)} chunk(s). "
                    f"Troncature silencieuse formellement interdite. Traitement suspendu."
                )
                logger.error(err_msg)
                for ov in overflow_details:
                    logger.error(f"  • {ov['chunk_id']} : {ov['token_count']} tokens (limite: {ov['token_limit']})")

                report.status = "halted_token_overflow"
                report.error_message = err_msg
                report.execution_time_seconds = time.perf_counter() - start_time
                return report, token_report

            # 3. Vectorisation des chunks (avec 'passage: ' en mémoire)
            ids, texts, embeddings, metadatas = self.embedding_manager.process_chunks(chunks)

            # 4. Stockage ChromaDB
            if mode == "rebuild":
                indexed_count, _ = self.chroma_store.rebuild_collection(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                indexed_count = self.chroma_store.upsert_chunks(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

            report.total_chunks_indexed = indexed_count
            report.status = "success"

        except Exception as e:
            logger.error(f"Erreur d'indexation : {e}")
            report.status = "error"
            report.error_message = str(e)
            token_report = TokenAuditReport()

        report.execution_time_seconds = time.perf_counter() - start_time
        return report, token_report

    @staticmethod
    def print_report(report: IndexingReport, token_report: TokenAuditReport) -> None:
        """Affiche le rapport complet d'indexation dans la console."""
        print("\n" + "=" * 88)
        print("RAPPORT D'INDEXATION VECTORIELLE CHROMADB — DATANOVA TECHNOLOGIES")
        print("=" * 88)
        print(f"Modèle d'embedding retenu            : {report.model_name}")
        print(f"Révision immuable (Commit SHA-1)     : {report.model_revision}")
        print(f"Dimension vectorielle exacte        : {report.embedding_dim}")
        print(f"Taille de batch CPU                 : {report.batch_size}")
        print(f"Collection cible ChromaDB           : {report.collection_name}")
        print(f"Répertoire persistant local         : {report.persist_directory}")
        print(f"Mode d'exécution                    : {report.mode.upper()}")
        print("-" * 88)
        print("AUDIT DE LONGUEUR TOKENISÉE (POLITIQUE ANTI-TRONCATURE) :")
        print(f"  • Chunks inspectés                : {token_report.total_chunks}")
        print(f"  • Budget maximal disponible       : {token_report.max_token_limit} tokens")
        print(f"  • Longueur minimale               : {token_report.min_tokens} tokens")
        print(f"  • Longueur maximale observée      : {token_report.max_tokens} tokens")
        print(f"  • Longueur moyenne                : {token_report.avg_tokens} tokens")
        print(f"  • Chunks en dépassement (overflow): {token_report.overflow_chunks_count}")

        if token_report.has_overflow:
            print("\nDétail des chunks en dépassement :")
            for ov in token_report.overflow_chunks_details:
                print(f"  [ALERTE] {ov['chunk_id']} ({ov['doc_type']}) : {ov['token_count']} tokens (+{ov['overflow']})")

        print("-" * 88)
        print("STATUT ET CARDINALITÉ DE L'INDEXATION :")
        print(f"  • Chunks chargés en mémoire       : {report.total_chunks_loaded}")
        print(f"  • Chunks indexés dans ChromaDB    : {report.total_chunks_indexed}")
        print(f"  • Statut final d'exécution        : {report.status.upper()}")
        if report.error_message:
            print(f"  • Message d'erreur                : {report.error_message}")

        print("-" * 88)
        print(f"Durée réelle d'exécution mesurée    : {report.execution_time_seconds:.3f} seconde(s)")
        print("=" * 88 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = IndexationPipeline()
    rep, tok_rep = pipeline.run(mode="rebuild")
    IndexationPipeline.print_report(rep, tok_rep)

