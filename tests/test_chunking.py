"""Tests unitaires et de non-régression pour le module de chunking documentaire."""
import pytest
from pathlib import Path

from src.ingestion.models import NormalizedDocument
from src.chunking.models import ChunkDocument, ChunkValidationReport
from src.chunking.chunking_config import ChunkingConfig, ChunkingStrategy
from src.chunking.text_splitter import (
    FixedTextSplitter,
    StructuralTextSplitter,
    generate_chunk_id
)
from src.chunking.chunk_validator import ChunkValidator
from src.chunking.chunking_manager import ChunkingManager


# ==============================================================================
# FIXTURES DE DOCUMENTS NORMALISÉS CONTRÔLÉS
# ==============================================================================

@pytest.fixture
def sample_faq_doc() -> NormalizedDocument:
    return NormalizedDocument(
        id="FAQ-TEST-001",
        page_content="Question : Comment configurer Delta Lake sur le cluster Spark ?\nRéponse : Définir spark.sql.extensions sur io.delta.sql.DeltaSparkSessionExtension.",
        metadata={
            "document_id": "DOC-FAQ-001",
            "source": "data/raw/faq/technical_faq.json",
            "date_version": "1.5.0",
            "doc_type": "faq",
            "sensitivity_level": 2,
            "faq_id": "FAQ-TEST-001",
            "category": "Data Engineering"
        }
    )


@pytest.fixture
def sample_csv_doc() -> NormalizedDocument:
    content = (
        "Dataset : Transactions Bancaires (Identifiant : DS-001)\n"
        "Description : Données transactionnelles brutes ingérées en streaming.\n"
        "Domaine d'activité : Finance | Équipe responsable : Équipe Data Platform\n"
        "Stockage Lakehouse : Couche Bronze (Format de stockage : Parquet/Snappy)\n"
        "Fréquence de mise à jour : Temps réel (micro-batch 1 min) | Rétention des données : 5 ans\n"
        "Règles et contrôles qualité : Vérification unicité transaction_id, non-nullité montant et timestamp ISO-8601\n"
        "Politique et habilitations d'accès : Accès restreint équipe Finance et Fraude (Niveau de sensibilité : 3)"
    )
    return NormalizedDocument(
        id="DS-001",
        page_content=content,
        metadata={
            "document_id": "DOC-CAT-001",
            "source": "data/raw/csv/dataset_catalog.csv",
            "date_version": "2.3.0",
            "doc_type": "csv",
            "sensitivity_level": 3,
            "dataset_id": "DS-001",
            "domain": "Finance",
            "owner_team": "Équipe Data Platform"
        }
    )


@pytest.fixture
def sample_pdf_page_doc() -> NormalizedDocument:
    # Page de ~800 caractères
    paragraphs = [
        "Section 1 : Principes généraux de la gouvernance des données chez DataNova Technologies.",
        "Toutes les données ingérées dans la plateforme doivent obligatoirement être étiquetées avec un niveau d'accréditation RBAC.",
        "Le chiffrement au repos via AES-256 et le chiffrement en transit via TLS 1.3 sont appliqués sans exception sur tous les buckets S3.",
        "Les équipes opérationnelles doivent consigner tout incident de sécurité dans un délai maximal de 15 minutes."
    ]
    return NormalizedDocument(
        id="DOC-SEC-001#p1",
        page_content="\n\n".join(paragraphs),
        metadata={
            "document_id": "DOC-SEC-001",
            "source": "data/raw/pdf/data_governance_security_policy.pdf",
            "date_version": "1.2.0",
            "doc_type": "pdf",
            "sensitivity_level": 3,
            "page_number": 1,
            "total_pages": 3
        }
    )


@pytest.fixture
def sample_markdown_doc() -> NormalizedDocument:
    md_content = """# Architecture Globale DataNova

## 1. Contexte Stratégique
La plateforme DataNova permet le traitement unifié des flux analytiques et IA.
Les micro-services communiquent par Kafka et les tables sont formatées en Delta Lake.

## 2. Ingestion et Streaming
Le pipeline d'ingestion supporte le CDC depuis les bases PostgreSQL vers Kafka Connect.
Chaque message est validé par le Schema Registry Confluent avant diffusion.

## 3. Sécurité et Contrôle d'Accès
L'authentification passe par OAuth2 et les rôles RBAC sont propagés aux requêtes RAG.
"""
    return NormalizedDocument(
        id="DOC-ARCH-001",
        page_content=md_content,
        metadata={
            "document_id": "DOC-ARCH-001",
            "source": "data/raw/markdown/data_platform_architecture.md",
            "date_version": "2.1.0",
            "doc_type": "markdown",
            "sensitivity_level": 1,
            "title": "Architecture Globale"
        }
    )


# ==============================================================================
# TESTS PYTEST
# ==============================================================================

def test_fixed_chunking_splitting(sample_pdf_page_doc):
    """1. Vérifie le découpage fixe régulier et la cohérence des index/totaux."""
    config = ChunkingConfig.create_fixed(chunk_size=200, chunk_overlap=30)
    splitter = FixedTextSplitter(config)

    chunks = splitter.split_document(sample_pdf_page_doc)
    assert len(chunks) > 1, "Un texte long doit produire plusieurs chunks en découpage fixe."

    total = len(chunks)
    for idx, c in enumerate(chunks):
        assert c.chunking_strategy == "fixed"
        assert c.chunk_index == idx
        assert c.total_chunks == total
        assert "c" in c.chunk_id
        assert c.sensitivity_level == 3
        assert c.document_id == "DOC-SEC-001"


def test_structural_markdown_sections(sample_markdown_doc):
    """2. Vérifie la préservation des sections Markdown cohérentes selon les titres ##."""
    config = ChunkingConfig.create_structural(markdown_max_section_size=1200)
    splitter = StructuralTextSplitter(config)

    chunks = splitter.split_document(sample_markdown_doc)
    assert len(chunks) >= 3, "Chaque section majeure doit générer au moins un chunk."

    sections_found = [c.metadata.get("section") for c in chunks]
    assert any("Contexte Stratégique" in str(s) for s in sections_found)
    assert any("Ingestion et Streaming" in str(s) for s in sections_found)
    assert any("Sécurité et Contrôle d'Accès" in str(s) for s in sections_found)

    for c in chunks:
        assert c.chunking_strategy == "structural"
        assert c.doc_type == "markdown"
        assert c.document_id == "DOC-ARCH-001"


def test_structural_pdf_page_boundaries(sample_pdf_page_doc):
    """3. Vérifie le respect strict des frontières de page pour les PDF."""
    config = ChunkingConfig.create_structural(chunk_size=300, chunk_overlap=30)
    splitter = StructuralTextSplitter(config)

    chunks = splitter.split_document(sample_pdf_page_doc)
    assert len(chunks) >= 1

    for c in chunks:
        assert c.metadata.get("page_number") == 1
        assert c.metadata.get("total_pages") == 3
        assert c.document_id == "DOC-SEC-001"
        assert c.sensitivity_level == 3


def test_structural_faq_integrity(sample_faq_doc):
    """4. Vérifie que la paire Question / Réponse est conservée entière dans un seul chunk unitaire."""
    config = ChunkingConfig.create_structural()
    splitter = StructuralTextSplitter(config)

    chunks = splitter.split_document(sample_faq_doc)
    assert len(chunks) == 1, "L'entrée FAQ doit rester atomique (1 seul chunk)."

    chunk = chunks[0]
    assert "Question :" in chunk.content
    assert "Réponse :" in chunk.content
    assert chunk.metadata.get("faq_id") == "FAQ-TEST-001"
    assert chunk.metadata.get("category") == "Data Engineering"
    assert chunk.total_chunks == 1
    assert chunk.chunk_index == 0


def test_structural_csv_integrity(sample_csv_doc):
    """5. Vérifie que la fiche dataset CSV (12 dimensions) est conservée unitaire."""
    config = ChunkingConfig.create_structural()
    splitter = StructuralTextSplitter(config)

    chunks = splitter.split_document(sample_csv_doc)
    assert len(chunks) == 1, "La fiche dataset CSV doit rester atomique (1 seul chunk)."

    chunk = chunks[0]
    assert "Dataset : Transactions Bancaires" in chunk.content
    assert "Niveau de sensibilité : 3" in chunk.content
    assert chunk.metadata.get("dataset_id") == "DS-001"
    assert chunk.metadata.get("domain") == "Finance"
    assert chunk.sensitivity_level == 3


def test_short_documents_preservation(sample_faq_doc):
    """6. Vérifie que les documents courts ne sont pas altérés ou tronqués."""
    config = ChunkingConfig.create_fixed(chunk_size=1000, chunk_overlap=50)
    splitter = FixedTextSplitter(config)

    chunks = splitter.split_document(sample_faq_doc)
    assert len(chunks) == 1
    assert chunks[0].content == sample_faq_doc.page_content


def test_long_documents_splitting():
    """7. Vérifie le bon découpage récursif d'un document volumineux sans perte de contenu."""
    long_text = "\n\n".join([f"Paragraphe numéro {i} contenant des informations techniques importantes." for i in range(50)])
    doc = NormalizedDocument(
        id="DOC-LONG-001",
        page_content=long_text,
        metadata={
            "document_id": "DOC-LONG-001",
            "source": "data/raw/long.md",
            "date_version": "1.0",
            "doc_type": "markdown",
            "sensitivity_level": 2
        }
    )
    config = ChunkingConfig.create_fixed(chunk_size=400, chunk_overlap=40)
    splitter = FixedTextSplitter(config)

    chunks = splitter.split_document(doc)
    assert len(chunks) > 5
    assert all(len(c.content) > 0 for c in chunks)


def test_empty_content_handling():
    """8. Vérifie la gestion robuste des documents vides."""
    # Création d'un NormalizedDocument avec espace blanc (doit produire 0 chunks sans planter)
    doc_whitespace = NormalizedDocument(
        id="DOC-EMPTY-001",
        page_content="   \n\t  ",
        metadata={
            "document_id": "DOC-EMPTY-001",
            "source": "data/raw/empty.md",
            "date_version": "1.0",
            "doc_type": "markdown",
            "sensitivity_level": 1
        }
    )
    splitter_fix = FixedTextSplitter(ChunkingConfig.create_fixed())
    splitter_str = StructuralTextSplitter(ChunkingConfig.create_structural())

    assert splitter_fix.split_document(doc_whitespace) == []
    assert splitter_str.split_document(doc_whitespace) == []


def test_metadata_and_provenance_preservation(sample_csv_doc):
    """9. Vérifie la conservation intégrale des 5 métadonnées obligatoires et champs spécifiques."""
    config = ChunkingConfig.create_structural()
    splitter = StructuralTextSplitter(config)

    chunks = splitter.split_document(sample_csv_doc)
    c = chunks[0]

    # 5 métadonnées obligatoires
    assert c.document_id == "DOC-CAT-001"
    assert c.source == "data/raw/csv/dataset_catalog.csv"
    assert c.date_version == "2.3.0"
    assert c.doc_type == "csv"
    assert c.sensitivity_level == 3

    # Métadonnées de chunking
    assert c.chunk_id is not None
    assert c.chunk_index == 0
    assert c.total_chunks == 1
    assert c.chunking_strategy == "structural"

    # Métadonnées spécifiques
    assert c.metadata.get("dataset_id") == "DS-001"
    assert c.metadata.get("domain") == "Finance"


def test_deterministic_and_unique_chunk_ids(sample_pdf_page_doc):
    """10. Vérifie que les identifiants sont strictement stables et exempts de collision."""
    config = ChunkingConfig.create_structural()
    splitter = StructuralTextSplitter(config)

    run_1 = splitter.split_document(sample_pdf_page_doc)
    run_2 = splitter.split_document(sample_pdf_page_doc)

    ids_1 = [c.chunk_id for c in run_1]
    ids_2 = [c.chunk_id for c in run_2]

    # Stabilité et déterminisme parfait
    assert ids_1 == ids_2

    # Unicité au sein du run
    assert len(ids_1) == len(set(ids_1))

    # Différentiation selon la stratégie
    splitter_fix = FixedTextSplitter(ChunkingConfig.create_fixed())
    run_fix = splitter_fix.split_document(sample_pdf_page_doc)
    ids_fix = [c.chunk_id for c in run_fix]
    assert set(ids_1).isdisjoint(set(ids_fix))


def test_chunk_validator_rejections_and_warnings(sample_faq_doc):
    """11. Vérifie le rejet des chunks invalides et la génération d'avertissements informatifs."""
    validator = ChunkValidator(min_chunk_size=50, soft_max_size=300)

    # Chunk valide
    valid_chunk = ChunkDocument(
        chunk_id="VALID_001",
        content="Contenu de test de longueur normale et bien formaté.",
        metadata={
            "document_id": "DOC-001",
            "source": "data/raw/test.md",
            "date_version": "1.0",
            "doc_type": "markdown",
            "sensitivity_level": 2,
            "chunk_id": "VALID_001",
            "chunk_index": 0,
            "total_chunks": 1,
            "chunking_strategy": "fixed"
        }
    )
    is_valid, reason, warnings = validator.validate_chunk(valid_chunk)
    assert is_valid is True
    assert reason is None

    # Rejet 1 : contenu vide
    invalid_content = ChunkDocument(
        chunk_id="INV_001",
        content="   ",
        metadata=dict(valid_chunk.metadata)
    )
    is_v, reas, _ = validator.validate_chunk(invalid_content)
    assert is_v is False
    assert "vide" in reas

    # Rejet 2 : niveau de sensibilité hors bornes
    invalid_sens = ChunkDocument(
        chunk_id="INV_002",
        content="Texte valide",
        metadata={**valid_chunk.metadata, "sensitivity_level": 5}
    )
    is_v, reas, _ = validator.validate_chunk(invalid_sens)
    assert is_v is False
    assert "sensibilité" in reas

    # Rejet 3 : métadonnée obligatoire manquante
    missing_meta = dict(valid_chunk.metadata)
    del missing_meta["source"]
    invalid_meta = ChunkDocument(
        chunk_id="INV_003",
        content="Texte valide",
        metadata=missing_meta
    )
    is_v, reas, _ = validator.validate_chunk(invalid_meta)
    assert is_v is False
    assert "source" in reas

    # Rejet 4 : fragment purement décoratif (séparateur horizontal '---')
    invalid_separator = ChunkDocument(
        chunk_id="INV_004",
        content="---",
        metadata=dict(valid_chunk.metadata)
    )
    is_v, reas, _ = validator.validate_chunk(invalid_separator)
    assert is_v is False
    assert "séparateur" in reas

    # Validation en lot avec collision d'ID
    _, report = validator.validate_batch([valid_chunk, valid_chunk])
    assert report.rejected_chunks_count == 1
    assert "Collision" in report.rejected_details[0]["reason"]


def test_comparison_report_generation(sample_pdf_page_doc, sample_faq_doc, sample_csv_doc, sample_markdown_doc):
    """12. Vérifie la génération correcte du rapport comparatif et le calcul des métriques."""
    docs = [sample_pdf_page_doc, sample_faq_doc, sample_csv_doc, sample_markdown_doc]
    manager = ChunkingManager()

    report, chunks_by_strat = manager.compare_strategies(docs)

    assert "fixed" in report.configs
    assert "structural" in report.configs

    fixed_stats = report.configs["fixed"]
    struct_stats = report.configs["structural"]

    assert fixed_stats.total_documents_input == 4
    assert struct_stats.total_documents_input == 4
    assert fixed_stats.total_chunks_output > 0
    assert struct_stats.total_chunks_output > 0
    assert fixed_stats.avg_chunk_length > 0
    assert struct_stats.avg_chunk_length > 0
    assert fixed_stats.metadata_preservation_rate == 100.0
    assert struct_stats.metadata_preservation_rate == 100.0


def test_save_and_load_chunks(tmp_path, sample_faq_doc):
    """Vérifie la sérialisation JSON reproductible (séparation métadonnées / chunks) et son chargement."""
    manager = ChunkingManager(output_dir=tmp_path)
    cfg = ChunkingConfig.create_structural()
    splitter = StructuralTextSplitter(cfg)

    chunks = splitter.split_document(sample_faq_doc)
    saved_file = manager.save_chunks(chunks, "test_chunks.json", cfg)

    assert saved_file.exists()

    loaded_chunks = ChunkingManager.load_chunks(saved_file)
    assert len(loaded_chunks) == len(chunks)
    assert loaded_chunks[0].chunk_id == chunks[0].chunk_id
    assert loaded_chunks[0].content == chunks[0].content
    assert loaded_chunks[0].metadata == chunks[0].metadata

