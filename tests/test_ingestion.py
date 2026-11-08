"""Suite de tests unitaires pour le pipeline d'ingestion documentaire multi-sources."""
import pytest
import json
from pathlib import Path
from typing import List

from src.ingestion.models import NormalizedDocument, IngestionReport
from src.ingestion.metadata_manager import MetadataManager, MetadataDiscrepancy
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.markdown_loader import MarkdownLoader
from src.ingestion.json_loader import JSONLoader
from src.ingestion.csv_loader import CSVLoader
from src.ingestion.document_validator import DocumentValidator
from src.ingestion.loader_factory import LoaderFactory, UnsupportedFormatError
from src.ingestion.ingestion_pipeline import IngestionPipeline


@pytest.fixture(scope="module")
def metadata_manager() -> MetadataManager:
    """Fixture fournissant une instance configurée de MetadataManager sur le corpus actif."""
    return MetadataManager(Path("data/raw/inventory.json"))


@pytest.fixture(scope="module")
def pipeline() -> IngestionPipeline:
    """Fixture fournissant l'orchestrateur d'ingestion configuré."""
    return IngestionPipeline(
        raw_data_dir=Path("data/raw"),
        inventory_path=Path("data/raw/inventory.json")
    )


# ==============================================================================
# 1. Test du chargement des PDF
# ==============================================================================
def test_pdf_loader_loading_and_pages(metadata_manager: MetadataManager):
    """Vérifie le chargement des PDF page par page, l'extraction de texte et les métadonnées."""
    loader = PDFLoader(metadata_manager)
    pdf_path = Path("data/raw/pdf/data_governance_security_policy.pdf")
    docs, ignored = loader.load(pdf_path)

    assert len(docs) == 3, "Le PDF DOC-SEC-001 doit produire exactement 3 documents (1 par page)."
    assert len(ignored) == 0, "Aucune page ne doit être ignorée sur ce fichier nominal."
    
    # Vérification des identifiants et métadonnées spécifiques
    for idx, doc in enumerate(docs, start=1):
        assert doc.id == f"DOC-SEC-001#p{idx}"
        assert doc.document_id == "DOC-SEC-001"
        assert doc.doc_type == "pdf"
        assert doc.sensitivity_level == 2
        assert doc.metadata["page_number"] == idx
        assert doc.metadata["total_pages"] == 3
        assert len(doc.page_content) > 500, "Le texte extrait doit être substantiel."


# ==============================================================================
# 2. Test du chargement des Markdown
# ==============================================================================
def test_markdown_loader_structure_and_frontmatter(metadata_manager: MetadataManager):
    """Vérifie le chargement du Markdown, le parsing du frontmatter et la préservation de la structure."""
    loader = MarkdownLoader(metadata_manager)
    md_path = Path("data/raw/markdown/data_platform_architecture.md")
    docs = loader.load(md_path)

    assert len(docs) == 1, "Le fichier Markdown doit produire 1 document normalisé."
    doc = docs[0]
    assert doc.id == "DOC-ARCH-001"
    assert doc.document_id == "DOC-ARCH-001"
    assert doc.doc_type == "markdown"
    assert doc.sensitivity_level == 2
    assert "Architecture de Référence" in doc.page_content
    assert "```mermaid" in doc.page_content, "Les blocs de code et diagrammes doivent être préservés."


# ==============================================================================
# 3. Test du chargement des FAQ JSON
# ==============================================================================
def test_json_loader_faq_documents(metadata_manager: MetadataManager):
    """Vérifie que les 32 questions de la FAQ technique deviennent 32 documents indépendants."""
    loader = JSONLoader(metadata_manager)
    faq_path = Path("data/raw/faq/technical_faq.json")
    docs = loader.load(faq_path)

    assert len(docs) == 32, "Le fichier technical_faq.json doit produire exactement 32 documents."
    
    first = docs[0]
    assert first.id == "FAQ-DE-001"
    assert first.document_id == "DOC-FAQ-001"
    assert first.doc_type == "faq"
    assert first.metadata["faq_id"] == "FAQ-DE-001"
    assert "category" in first.metadata
    assert first.page_content.startswith("Question :")
    assert "Réponse :" in first.page_content


# ==============================================================================
# 4. Test du chargement du CSV
# ==============================================================================
def test_csv_loader_dataset_catalog(metadata_manager: MetadataManager):
    """Vérifie que les 22 datasets du CSV deviennent 22 documents avec leurs 12 colonnes."""
    loader = CSVLoader(metadata_manager)
    csv_path = Path("data/raw/csv/dataset_catalog.csv")
    docs = loader.load(csv_path)

    assert len(docs) == 22, "Le catalogue CSV doit produire exactement 22 documents."
    
    first = docs[0]
    assert first.id == "DS-001"
    assert first.document_id == "DOC-CAT-001"
    assert first.doc_type == "csv"
    assert first.metadata["dataset_id"] == "DS-001"
    assert first.metadata["domain"] == "Telemetry"
    assert "storage_layer" in first.metadata
    assert "storage_format" in first.metadata
    assert "access_policy" in first.metadata
    assert "Dataset :" in first.page_content
    assert "Règles et contrôles qualité :" in first.page_content


# ==============================================================================
# 5. Test de la qualité d'extraction du contenu textuel
# ==============================================================================
def test_content_extraction_quality(pipeline: IngestionPipeline):
    """Vérifie qu'aucun document n'a un contenu vide, tronqué ou constitué uniquement d'espaces."""
    docs, _ = pipeline.run()
    assert len(docs) > 0

    for doc in docs:
        assert isinstance(doc.page_content, str)
        clean = doc.page_content.strip()
        assert len(clean) >= 20, f"Le document '{doc.id}' a un contenu anormalement court ({len(clean)} car.)."


# ==============================================================================
# 6. Test de la présence obligatoire des 5 métadonnées universelles
# ==============================================================================
def test_mandatory_metadata_presence(pipeline: IngestionPipeline):
    """Vérifie que 100% des documents possèdent les 5 métadonnées universelles obligatoires."""
    docs, _ = pipeline.run()
    mandatory_fields = ["document_id", "source", "date_version", "doc_type", "sensitivity_level"]

    for doc in docs:
        meta = doc.metadata
        for f in mandatory_fields:
            assert f in meta, f"Champ obligatoire '{f}' absent du document '{doc.id}'."
            assert meta[f] is not None and str(meta[f]).strip() != "", f"Champ '{f}' vide sur '{doc.id}'."


# ==============================================================================
# 7. Test d'unicité et de stabilité des identifiants
# ==============================================================================
def test_document_id_uniqueness_and_stability(pipeline: IngestionPipeline):
    """Vérifie l'unicité stricte de chaque identifiant 'id' dans toute la collection."""
    docs_run1, _ = pipeline.run()
    docs_run2, _ = pipeline.run()

    ids_run1 = [d.id for d in docs_run1]
    ids_run2 = [d.id for d in docs_run2]

    # Unicité au sein d'une même exécution
    assert len(ids_run1) == len(set(ids_run1)), "Des identifiants 'id' en doublon ont été détectés."
    
    # Stabilité et reproductibilité entre deux exécutions
    assert ids_run1 == ids_run2, "Les identifiants générés doivent être déterministes et stables."


# ==============================================================================
# 8. Test de la validité des niveaux de sensibilité [1..3]
# ==============================================================================
def test_sensitivity_levels_bounds(pipeline: IngestionPipeline):
    """Vérifie que tous les niveaux de sensibilité sont strictement des entiers valides (1, 2 ou 3)."""
    docs, _ = pipeline.run()

    for doc in docs:
        lvl = doc.sensitivity_level
        assert isinstance(lvl, int), f"Le niveau de sensibilité de '{doc.id}' doit être un entier."
        assert lvl in (1, 2, 3), f"Niveau de sensibilité {lvl} hors bornes pour '{doc.id}'."


# ==============================================================================
# 9. Test du traitement des fichiers vides ou invalides
# ==============================================================================
def test_corrupted_or_empty_file_handling(tmp_path: Path, metadata_manager: MetadataManager):
    """Vérifie le comportement robuste du système face à des fichiers vides ou malformés."""
    # Test Markdown vide
    empty_md = tmp_path / "empty.md"
    empty_md.write_text("   \n\n  ", encoding="utf-8")
    md_loader = MarkdownLoader(metadata_manager)
    with pytest.raises(ValueError, match="vide"):
        md_loader.load(empty_md)

    # Test JSON FAQ vide
    empty_json = tmp_path / "empty_faq.json"
    empty_json.write_text("[]", encoding="utf-8")
    json_loader = JSONLoader(metadata_manager)
    with pytest.raises(ValueError, match="vide"):
        json_loader.load(empty_json)

    # Test DocumentValidator sur document sans contenu
    bad_doc = NormalizedDocument(
        id="BAD-01",
        page_content="   ",
        metadata={"document_id": "B1", "source": "s", "date_version": "1.0", "doc_type": "pdf", "sensitivity_level": 1, "page_number": 1}
    )
    is_valid, error, _ = DocumentValidator.validate(bad_doc)
    assert not is_valid
    assert "vide" in error.lower()


# ==============================================================================
# 10. Test de l'exclusion stricte des fixtures de test
# ==============================================================================
def test_strict_fixtures_exclusion(pipeline: IngestionPipeline):
    """Vérifie qu'aucun fichier de data/test_fixtures/ ne pénètre dans la collection active."""
    docs, report = pipeline.run()

    for doc in docs:
        src = doc.source.lower()
        assert "test_fixtures" not in src, f"Document fixture détecté illégalement : {src}"
        assert "outdated" not in src, f"Document obsolète détecté illégalement : {src}"
        assert "conflicting" not in src, f"Document contradictoire détecté illégalement : {src}"
        assert "adversarial" not in src, f"Document d'attaque détecté illégalement : {src}"


# ==============================================================================
# 11. Test de la résilience du pipeline en cas d'erreur sur un fichier isolé
# ==============================================================================
def test_pipeline_resilience_on_single_file_error(tmp_path: Path):
    """Vérifie qu'une erreur sur un fichier corrompu n'arrête pas l'ingestion des autres fichiers."""
    raw_copy = tmp_path / "raw"
    raw_copy.mkdir()
    
    # Création d'un markdown valide
    valid_md = raw_copy / "valid.md"
    valid_md.write_text("# Titre Valide\n\nContenu valide et substantiel pour le test.", encoding="utf-8")

    # Création d'un fichier corrompu avec extension .pdf
    corrupt_pdf = raw_copy / "corrupted.pdf"
    corrupt_pdf.write_bytes(b"NOT_A_REAL_PDF_HEADER_DATA_CORRUPTION")

    # Inventaire minimaliste
    inv_file = raw_copy / "inventory.json"
    inv_file.write_text(json.dumps([
        {"id": "DOC-OK", "path": "valid.md", "title": "Doc Valide", "type": "markdown", "version": "1.0", "sensitivity_level": 1, "owner_team": "DE", "is_active": True},
        {"id": "DOC-KO", "path": "corrupted.pdf", "title": "Doc Corrompu", "type": "pdf", "version": "1.0", "sensitivity_level": 2, "owner_team": "IT", "is_active": True}
    ]), encoding="utf-8")

    pipe = IngestionPipeline(raw_data_dir=raw_copy, inventory_path=inv_file)
    docs, report = pipe.run()

    # Le fichier valide doit être ingéré malgré l'échec du fichier corrompu
    assert len(docs) == 1
    assert docs[0].id == "DOC-OK"
    assert report.errors_count == 1, "L'erreur sur le fichier corrompu doit être consignée."
    assert any("corrupted.pdf" in err for err in report.errors)


# ==============================================================================
# 12. Test de la compatibilité stricte ChromaDB (types primitifs)
# ==============================================================================
def test_chromadb_compatibility(pipeline: IngestionPipeline):
    """Vérifie que 100% des métadonnées de tous les documents sont de types primitifs autorisés."""
    docs, _ = pipeline.run()

    for doc in docs:
        for key, val in doc.metadata.items():
            assert isinstance(val, (str, int, float, bool)), (
                f"La métadonnée '{key}' du document '{doc.id}' a un type complexe '{type(val).__name__}' "
                f"incompatible avec ChromaDB."
            )
