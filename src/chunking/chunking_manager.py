"""Gestionnaire et orchestrateur du découpage documentaire (chunking)."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.ingestion.models import NormalizedDocument
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.chunking.models import (
    ChunkDocument,
    ChunkValidationReport,
    ChunkingStats,
    ChunkComparisonReport
)
from src.chunking.chunking_config import ChunkingConfig, ChunkingStrategy
from src.chunking.text_splitter import FixedTextSplitter, StructuralTextSplitter
from src.chunking.chunk_validator import ChunkValidator

logger = logging.getLogger(__name__)


class ChunkingManager:
    """
    Orchestrateur central du chunking pour le projet RAG Multi-Sources de DataNova Technologies.
    
    Responsabilités :
      1. Traiter une collection de NormalizedDocument en mémoire selon la stratégie spécifiée.
      2. Contrôler chaque lot généré via ChunkValidator (sécurité RBAC, provenance, conformité ChromaDB).
      3. Calculer les métriques descriptives sans assimiler la taille cible à une limite absolue rigide.
      4. Séparer rigoureusement les métadonnées d'exécution (timestamp, config) des données des chunks.
      5. Sauvegarder les chunks de façon déterministe et reproductible au format JSON.
      6. Comparer les deux configurations (Fixe vs Structurelle) sur un corpus identique.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("data/processed")

    def process_documents(
        self,
        documents: List[NormalizedDocument],
        config: ChunkingConfig
    ) -> Tuple[List[ChunkDocument], ChunkValidationReport, ChunkingStats]:
        """
        Découpe et valide une liste de NormalizedDocument selon la configuration fournie.
        """
        # Sélection du découpeur
        if config.strategy == ChunkingStrategy.STRUCTURAL.value:
            splitter = StructuralTextSplitter(config)
        else:
            splitter = FixedTextSplitter(config)

        # 1. Découpage
        raw_chunks = splitter.split_documents(documents)

        # 2. Validation
        validator = ChunkValidator(
            min_chunk_size=config.min_chunk_size,
            soft_max_size=config.soft_max_size
        )
        valid_chunks, validation_report = validator.validate_batch(raw_chunks)

        # 3. Calcul des statistiques descriptives
        stats = self._compute_stats(documents, valid_chunks, config)

        return valid_chunks, validation_report, stats

    def _compute_stats(
        self,
        documents: List[NormalizedDocument],
        chunks: List[ChunkDocument],
        config: ChunkingConfig
    ) -> ChunkingStats:
        """Calcule les statistiques d'un lot de chunks validés."""
        total_chunks = len(chunks)
        if total_chunks == 0:
            return ChunkingStats(
                strategy_name=config.strategy,
                total_documents_input=len(documents),
                total_chunks_output=0
            )

        lengths = [len(c.content) for c in chunks]
        avg_len = sum(lengths) / total_chunks
        min_len = min(lengths)
        max_len = max(lengths)

        # Comptages informatifs basés sur les seuils configurés
        short_count = sum(1 for ln in lengths if ln < config.min_chunk_size)
        exceeded_count = sum(1 for ln in lengths if ln > config.soft_max_size)

        # Répartition RBAC et formats
        sens_dist: Dict[int, int] = {}
        fmt_dist: Dict[str, int] = {}
        for c in chunks:
            s = c.sensitivity_level
            sens_dist[s] = sens_dist.get(s, 0) + 1
            f = c.doc_type
            fmt_dist[f] = fmt_dist.get(f, 0) + 1

        return ChunkingStats(
            strategy_name=config.strategy,
            total_documents_input=len(documents),
            total_chunks_output=total_chunks,
            avg_chunk_length=round(avg_len, 1),
            min_chunk_length=min_len,
            max_chunk_length=max_len,
            short_chunks_count=short_count,
            target_exceeded_count=exceeded_count,
            metadata_preservation_rate=100.0,
            chunks_by_sensitivity=sens_dist,
            chunks_by_format=fmt_dist
        )

    def compare_strategies(
        self,
        documents: List[NormalizedDocument],
        fixed_config: Optional[ChunkingConfig] = None,
        structural_config: Optional[ChunkingConfig] = None
    ) -> Tuple[ChunkComparisonReport, Dict[str, List[ChunkDocument]]]:
        """
        Exécute et compare les stratégies Fixe et Structurelle sur le même lot de documents.
        """
        cfg_fixed = fixed_config or ChunkingConfig.create_fixed()
        cfg_struct = structural_config or ChunkingConfig.create_structural()

        chunks_fixed, val_fixed, stats_fixed = self.process_documents(documents, cfg_fixed)
        chunks_struct, val_struct, stats_struct = self.process_documents(documents, cfg_struct)

        # Échantillons représentatifs (1 par format pour chaque stratégie)
        sample_chunks: Dict[str, List[Dict[str, Any]]] = {
            "fixed": self._extract_samples(chunks_fixed),
            "structural": self._extract_samples(chunks_struct)
        }

        report = ChunkComparisonReport(
            configs={
                "fixed": stats_fixed,
                "structural": stats_struct
            },
            sample_chunks=sample_chunks
        )

        chunks_by_strategy = {
            "fixed": chunks_fixed,
            "structural": chunks_struct
        }

        return report, chunks_by_strategy

    def _extract_samples(self, chunks: List[ChunkDocument]) -> List[Dict[str, Any]]:
        """Extrait un échantillon représentatif de chunk pour chaque format documentaire."""
        samples: List[Dict[str, Any]] = []
        seen_formats = set()

        for c in chunks:
            fmt = c.doc_type
            if fmt not in seen_formats:
                seen_formats.add(fmt)
                samples.append({
                    "doc_type": fmt,
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "sensitivity_level": c.sensitivity_level,
                    "length": len(c.content),
                    "preview": c.content[:160] + ("..." if len(c.content) > 160 else "")
                })
        return samples

    def save_chunks(
        self,
        chunks: List[ChunkDocument],
        filename: str,
        config: ChunkingConfig,
        stats: Optional[ChunkingStats] = None
    ) -> Path:
        """
        Sauvegarde les chunks et leurs métadonnées de configuration au format JSON.
        
        Séparation stricte :
          - 'configuration_metadata' : contient le timestamp d'exécution, la stratégie et les paramètres.
          - 'chunks' : liste des chunks déterministes (sans timestamp dans leurs métadonnées).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.output_dir / filename

        export_data = {
            "configuration_metadata": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "strategy": config.strategy,
                "target_chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "min_chunk_size": config.min_chunk_size,
                "soft_max_size": config.soft_max_size,
                "total_chunks_saved": len(chunks),
                "stats": stats.model_dump() if stats else {}
            },
            "chunks": [c.to_dict() for c in chunks]
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Sauvegarde de {len(chunks)} chunks dans '{file_path}'.")
        return file_path

    @staticmethod
    def load_chunks(file_path: Path) -> List[ChunkDocument]:
        """Charge des chunks précédemment sauvegardés depuis un fichier JSON."""
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier de chunks introuvable : {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_chunks = data.get("chunks", [])
        return [ChunkDocument(**item) for item in raw_chunks]

    @staticmethod
    def print_comparison_table(report: ChunkComparisonReport) -> None:
        """Affiche un tableau comparatif structuré des stratégies dans le terminal."""
        fixed_s = report.configs.get("fixed")
        struct_s = report.configs.get("structural")

        if not fixed_s or not struct_s:
            print("Données de comparaison incomplètes.")
            return

        print("\n" + "=" * 88)
        print("TABLEAU COMPARATIF DES STRATÉGIES DE CHUNKING — DATANOVA TECHNOLOGIES")
        print("=" * 88)
        header = f"{'Métrique descriptive':<40} | {'Configuration A (Fixe)':<20} | {'Configuration B (Structurelle)':<20}"
        print(header)
        print("-" * 88)

        rows = [
            ("Documents normalisés en entrée", str(fixed_s.total_documents_input), str(struct_s.total_documents_input)),
            ("Nombre total de chunks générés", str(fixed_s.total_chunks_output), str(struct_s.total_chunks_output)),
            ("Taille moyenne des chunks (caractères)", f"{fixed_s.avg_chunk_length:.1f}", f"{struct_s.avg_chunk_length:.1f}"),
            ("Taille minimale observée (caractères)", str(fixed_s.min_chunk_length), str(struct_s.min_chunk_length)),
            ("Taille maximale observée (caractères)", str(fixed_s.max_chunk_length), str(struct_s.max_chunk_length)),
            ("Chunks courts (< seuil min informatif)", str(fixed_s.short_chunks_count), str(struct_s.short_chunks_count)),
            ("Chunks dépassant cible (+ tolérance)", str(fixed_s.target_exceeded_count), str(struct_s.target_exceeded_count)),
            ("Taux de préservation des métadonnées", f"{fixed_s.metadata_preservation_rate:.1f}%", f"{struct_s.metadata_preservation_rate:.1f}%"),
        ]

        for metric, f_val, s_val in rows:
            print(f"{metric:<40} | {f_val:<20} | {s_val:<20}")

        print("-" * 88)
        print("Répartition des chunks par format documentaire :")
        all_formats = sorted(set(list(fixed_s.chunks_by_format.keys()) + list(struct_s.chunks_by_format.keys())))
        for fmt in all_formats:
            f_count = fixed_s.chunks_by_format.get(fmt, 0)
            s_count = struct_s.chunks_by_format.get(fmt, 0)
            print(f"  • Format {fmt.upper():<30} | {f_count:<20} | {s_count:<20}")

        print("-" * 88)
        print("Répartition par niveau de sensibilité RBAC :")
        for lvl in (1, 2, 3):
            label = {1: "Niveau 1 (Public / Interne)", 2: "Niveau 2 (Restreint)", 3: "Niveau 3 (Confidentiel)"}.get(lvl)
            f_c = fixed_s.chunks_by_sensitivity.get(lvl, 0)
            s_c = struct_s.chunks_by_sensitivity.get(lvl, 0)
            print(f"  • {label:<30} | {f_c:<20} | {s_c:<20}")

        print("-" * 88)
        print("Échantillons représentatifs par format :")
        for strat_name in ("fixed", "structural"):
            label = "Configuration A (Fixe)" if strat_name == "fixed" else "Configuration B (Structurelle)"
            print(f"\n[{label}]")
            samples = report.sample_chunks.get(strat_name, [])
            for s in samples:
                print(f"  • Type {s['doc_type'].upper():<10} | ID: {s['chunk_id']} ({s['length']} car.)")
                print(f"    Aperçu: \"{s['preview']}\"")

        print("=" * 88 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = IngestionPipeline()
    docs, ingestion_rep = pipeline.run()

    manager = ChunkingManager()
    cfg_fix = ChunkingConfig.create_fixed()
    cfg_str = ChunkingConfig.create_structural()

    comparison_report, chunks_by_strat = manager.compare_strategies(docs, cfg_fix, cfg_str)

    # Sauvegarde des deux résultats pour la Phase 6
    manager.save_chunks(
        chunks_by_strat["fixed"],
        "chunks_fixed.json",
        cfg_fix,
        comparison_report.configs["fixed"]
    )
    manager.save_chunks(
        chunks_by_strat["structural"],
        "chunks_structural.json",
        cfg_str,
        comparison_report.configs["structural"]
    )

    # Affichage du rapport comparatif
    ChunkingManager.print_comparison_table(comparison_report)

