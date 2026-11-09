"""Tests unitaires et de non-régression pour les embeddings et le vector store ChromaDB."""
import pytest
import math
import hashlib
from pathlib import Path
from typing import List

from src.chunking.models import ChunkDocument
from src.embeddings.embedding_config import EmbeddingConfig
from src.embeddings.tokenizer_inspector import TokenizerInspector, TokenAuditReport
from src.embeddings.embedding_validator import EmbeddingValidator
from src.embeddings.embedding_manager import EmbeddingManager
from src.embeddings.chroma_store import ChromaStore
from src.embeddings.indexation_pipeline import IndexationPipeline


# ==============================================================================
# HELPERS DE SIMULATION HORS-LIGNE (MOCK DÉTERMINISTE SANS RÉSEAU)
# ==============================================================================

def mock_embedding_fn_384(texts: List[str]) -> List[List[float]]:
    """
    Générateur d'embeddings simulé déterministe de dimension 384.
    Produit des vecteurs normalisés L2 sans télécharger de modèle Hugging Face.
    """
    vectors: List[List[float]] = []
    for text in texts:
        vec: List[float] = []
        for i in range(384):
            # Calcul pseudo-aléatoire mais 100% déterministe basé sur SHA-256
            h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).digest()
            val = (int.from_bytes(h[:4], "little") / 2**32) - 0.5
            vec.append(val)

        # Normalisation L2 (norme = 1.0)
        norm = math.sqrt(sum(x * x for x in vec))
        normalized = [x / norm for x in vec]
        vectors.append(normalized)
    return vectors


def mock_tokenizer_fn(text: str) -> int:
    """Simulateur de tokenizer rapide : ~1 token pour 4 caractères."""
    words = text.split()
    return max(1, int(len(words) * 1.3))


@pytest.fixture
def sample_chunk_structural_1() -> ChunkDocument:
    return ChunkDocument(
        chunk_id="DOC-SEC-001#p1_str_c0_test1",
        content="Politique de sécurité DataNova Technologies relative au chiffrement AES-256.",
        metadata={
            "document_id": "DOC-SEC-001",
            "source": "data/raw/pdf/data_governance_security_policy.pdf",
            "date_version": "1.2.0",
            "doc_type": "pdf",
            "sensitivity_level": 3,
            "chunk_id": "DOC-SEC-001#p1_str_c0_test1",
            "chunk_index": 0,
            "total_chunks": 2,
            "chunking_strategy": "structural",
            "page_number": 1,
            "total_pages": 3
        }
    )


@pytest.fixture
def sample_chunk_structural_2() -> ChunkDocument:
    return ChunkDocument(
        chunk_id="DOC-SEC-001#p1_str_c1_test2",
        content="Contrôles d'accès RBAC et habilitations applicables aux pipelines Big Data.",
        metadata={
            "document_id": "DOC-SEC-001",
            "source": "data/raw/pdf/data_governance_security_policy.pdf",
            "date_version": "1.2.0",
            "doc_type": "pdf",
            "sensitivity_level": 3,
            "chunk_id": "DOC-SEC-001#p1_str_c1_test2",
            "chunk_index": 1,
            "total_chunks": 2,
            "chunking_strategy": "structural",
            "page_number": 1,
            "total_pages": 3
        }
    )


# ==============================================================================
# TESTS UNITAIRES
# ==============================================================================

def test_token_length_inspection(sample_chunk_structural_1, sample_chunk_structural_2):
    """1. Vérifie le calcul des longueurs tokenisées et la production du rapport de distribution."""
    inspector = TokenizerInspector(tokenize_fn=mock_tokenizer_fn)
    report = inspector.inspect_chunks([sample_chunk_structural_1, sample_chunk_structural_2])

    assert report.total_chunks == 2
    assert report.max_token_limit == 507
    assert report.min_tokens > 0
    assert report.max_tokens >= report.min_tokens
    assert report.avg_tokens > 0
    assert report.overflow_chunks_count == 0
    assert report.has_overflow is False


def test_token_overflow_detection_and_halt(tmp_path):
    """2. Vérifie qu'un chunk dépassant 507 tokens est intercepté sans troncature silencieuse."""
    # Simulation d'une fonction de tokenisation renvoyant 600 tokens
    def overflow_tokenizer(text: str) -> int:
        return 600

    inspector = TokenizerInspector(tokenize_fn=overflow_tokenizer)
    doc_overflow = ChunkDocument(
        chunk_id="DOC-OVERFLOW-001",
        content="Texte anormalement long générant un dépassement de token.",
        metadata={
            "document_id": "DOC-TEST",
            "source": "test.md",
            "date_version": "1.0",
            "doc_type": "markdown",
            "sensitivity_level": 2,
            "chunk_id": "DOC-OVERFLOW-001",
            "chunk_index": 0,
            "total_chunks": 1,
            "chunking_strategy": "structural"
        }
    )

    report = inspector.inspect_chunks([doc_overflow])
    assert report.has_overflow is True
    assert report.overflow_chunks_count == 1
    assert report.overflow_chunks_details[0]["chunk_id"] == "DOC-OVERFLOW-001"
    assert report.overflow_chunks_details[0]["token_count"] == 600


def test_embedding_config_revision_and_prefixes():
    """3. Vérifie la configuration immuable : commit SHA-1, dim 384, batch=16 et préfixes E5."""
    cfg = EmbeddingConfig.create_structural_config()

    assert cfg.model_name == "intfloat/multilingual-e5-small"
    assert len(cfg.model_revision) == 40  # SHA-1 complet à 40 caractères
    assert cfg.embedding_dim == 384
    assert cfg.max_seq_length == 512
    assert cfg.max_content_tokens == 507
    assert cfg.document_prefix == "passage: "
    assert cfg.query_prefix == "query: "
    assert cfg.batch_size == 16
    assert cfg.collection_name == "datanova_corpus_structural"


def test_embedding_validator_dimensions_and_safety():
    """4. Valide un vecteur 384-d conforme et rejette dimensions erronées, NaN, Inf et vecteurs nuls."""
    validator = EmbeddingValidator(expected_dim=384)

    # Vecteur valide
    valid_vec = [0.1] * 384
    is_valid, reason = validator.validate_vector(valid_vec)
    assert is_valid is True
    assert reason is None

    # Rejet dimension erronée
    is_v, reas = validator.validate_vector([0.1] * 128)
    assert is_v is False
    assert "Dimension incorrecte" in reas

    # Rejet NaN
    nan_vec = [0.1] * 383 + [float("nan")]
    is_v, reas = validator.validate_vector(nan_vec)
    assert is_v is False
    assert "NaN" in reas

    # Rejet Inf
    inf_vec = [0.1] * 383 + [float("inf")]
    is_v, reas = validator.validate_vector(inf_vec)
    assert is_v is False
    assert "Inf" in reas

    # Rejet vecteur nul
    zero_vec = [0.0] * 384
    is_v, reas = validator.validate_vector(zero_vec)
    assert is_v is False
    assert "Vecteur nul" in reas


def test_passage_prefix_in_memory_only(sample_chunk_structural_1):
    """5. Vérifie que 'passage: ' est envoyé au modèle mais ABSENT du texte stocké."""
    captured_texts_for_model: List[str] = []

    def capturing_mock_embed(texts: List[str]) -> List[List[float]]:
        captured_texts_for_model.extend(texts)
        return mock_embedding_fn_384(texts)

    manager = EmbeddingManager(embedding_fn=capturing_mock_embed)
    ids, stored_texts, embs, metas = manager.process_chunks([sample_chunk_structural_1])

    # Le modèle tensoriel a bien reçu le préfixe E5
    assert len(captured_texts_for_model) == 1
    assert captured_texts_for_model[0].startswith("passage: ")

    # Le texte stocké pour ChromaDB conserve le texte d'origine PUR sans préfixe
    assert len(stored_texts) == 1
    assert not stored_texts[0].startswith("passage: ")
    assert stored_texts[0] == sample_chunk_structural_1.content


def test_query_prefix_contract():
    """6. Valide le contrat d'interface pour la Phase 7 : embed_query applique obligatoirement 'query: '."""
    captured_query: List[str] = []

    def capturing_mock(texts: List[str]) -> List[List[float]]:
        captured_query.extend(texts)
        return mock_embedding_fn_384(texts)

    manager = EmbeddingManager(embedding_fn=capturing_mock)
    query_vec = manager.embed_query("Quelles sont les règles de rétention ?")

    assert len(query_vec) == 384
    assert len(captured_query) == 1
    assert captured_query[0] == "query: Quelles sont les règles de rétention ?"


def test_chroma_store_metadata_none_omission(tmp_path):
    """7. Vérifie que les valeurs None sont omises sans invention de chaînes et que les listes sont rejetées."""
    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store = ChromaStore(cfg)

    raw_meta = {
        "document_id": "DOC-001",
        "sensitivity_level": 2,
        "valid_flag": True,
        "score": 0.95,
        "empty_field": None,  # Doit être omis
    }

    clean = store.clean_metadata(raw_meta)
    assert "empty_field" not in clean
    assert clean["document_id"] == "DOC-001"
    assert clean["sensitivity_level"] == 2
    assert clean["valid_flag"] is True

    # Rejet d'un type complexe
    with pytest.raises(ValueError, match="non primitif"):
        store.clean_metadata({"invalid_list": ["a", "b"]})


def test_chroma_store_upsert_idempotency(tmp_path, sample_chunk_structural_1):
    """8. Vérifie l'idempotence : réexécuter upsert ne duplique aucun document."""
    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store = ChromaStore(cfg)
    manager = EmbeddingManager(embedding_fn=mock_embedding_fn_384)

    ids, texts, embs, metas = manager.process_chunks([sample_chunk_structural_1])

    # Premier upsert
    store.upsert_chunks(ids, texts, embs, metas)
    assert store.count() == 1

    # Deuxième upsert identique
    store.upsert_chunks(ids, texts, embs, metas)
    assert store.count() == 1


def test_chroma_store_update_existing_chunk(tmp_path, sample_chunk_structural_1):
    """9. Vérifie qu'un chunk dont le texte a changé est mis à jour sans modifier le count()."""
    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store = ChromaStore(cfg)
    manager = EmbeddingManager(embedding_fn=mock_embedding_fn_384)

    ids, texts, embs, metas = manager.process_chunks([sample_chunk_structural_1])
    store.upsert_chunks(ids, texts, embs, metas)
    assert store.count() == 1

    # Modification du texte du chunk existant
    updated_texts = ["Contenu révisé avec mise à jour des politiques de sécurité."]
    updated_embs = manager.embed_texts(updated_texts)
    store.upsert_chunks(ids, updated_texts, updated_embs, metas)

    assert store.count() == 1
    fetched = store.get_chunk(sample_chunk_structural_1.chunk_id)
    assert fetched["document"] == updated_texts[0]


def test_chroma_store_rebuild_deletes_obsolete_chunks(
    tmp_path,
    sample_chunk_structural_1,
    sample_chunk_structural_2
):
    """10. Vérifie que la reconstruction élimine les chunks supprimés du fichier source."""
    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store = ChromaStore(cfg)
    manager = EmbeddingManager(embedding_fn=mock_embedding_fn_384)

    # Initialisation avec 2 chunks
    ids, texts, embs, metas = manager.process_chunks([sample_chunk_structural_1, sample_chunk_structural_2])
    store.rebuild_collection(ids, texts, embs, metas)
    assert store.count() == 2

    # Reconstruction avec uniquement le premier chunk (le chunk 2 est devenu obsolète)
    ids_v2, texts_v2, embs_v2, metas_v2 = manager.process_chunks([sample_chunk_structural_1])
    store.rebuild_collection(ids_v2, texts_v2, embs_v2, metas_v2)

    assert store.count() == 1
    assert store.get_chunk(sample_chunk_structural_1.chunk_id) is not None
    assert store.get_chunk(sample_chunk_structural_2.chunk_id) is None


def test_chroma_store_rebuild_failure_preserves_original(tmp_path, sample_chunk_structural_1):
    """11. Vérifie qu'un échec en cours de staging préserve l'ancienne collection intacte."""
    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store = ChromaStore(cfg)
    manager = EmbeddingManager(embedding_fn=mock_embedding_fn_384)

    # Collection initiale avec 1 chunk
    ids, texts, embs, metas = manager.process_chunks([sample_chunk_structural_1])
    store.rebuild_collection(ids, texts, embs, metas)
    assert store.count() == 1

    # Tentative de rebuild avec un lot incohérent provoquant une erreur avant la bascule
    with pytest.raises(Exception):
        store.rebuild_collection(
            ids=["FAKE_1"],
            documents=["doc"],
            embeddings=mock_embedding_fn_384(["doc"]),
            metadatas=[{"invalid_nested": {"dict": 1}}]  # Rejeté par clean_metadata
        )

    # L'ancienne collection est 100% intacte
    assert store.count() == 1
    assert store.get_chunk(sample_chunk_structural_1.chunk_id) is not None


def test_chroma_store_collection_isolation(
    tmp_path,
    sample_chunk_structural_1,
    sample_chunk_structural_2
):
    """12. Vérifie l'isolation stricte : reconstruire la collection structurelle ne touche pas la collection fixe."""
    cfg_struct = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    cfg_fixed = EmbeddingConfig.create_fixed_config(persist_directory=tmp_path)

    store_struct = ChromaStore(cfg_struct)
    store_fixed = ChromaStore(cfg_fixed)
    manager = EmbeddingManager(embedding_fn=mock_embedding_fn_384)

    # Indexation de 2 chunks dans la collection fixe
    ids_f, txt_f, emb_f, meta_f = manager.process_chunks([sample_chunk_structural_1, sample_chunk_structural_2])
    store_fixed.rebuild_collection(ids_f, txt_f, emb_f, meta_f)
    assert store_fixed.count() == 2

    # Reconstruction de la collection structurelle avec 1 seul chunk
    ids_s, txt_s, emb_s, meta_s = manager.process_chunks([sample_chunk_structural_1])
    store_struct.rebuild_collection(ids_s, txt_s, emb_s, meta_s)

    # Vérification d'isolation absolue
    assert store_struct.count() == 1
    assert store_fixed.count() == 2  # Inchangée


def test_chroma_store_persistence_reload(tmp_path, sample_chunk_structural_1):
    """13. Vérifie la persistance sur disque après fermeture et réouverture du client."""
    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store1 = ChromaStore(cfg)
    manager = EmbeddingManager(embedding_fn=mock_embedding_fn_384)

    ids, texts, embs, metas = manager.process_chunks([sample_chunk_structural_1])
    store1.upsert_chunks(ids, texts, embs, metas)
    assert store1.count() == 1

    # Réouverture d'une nouvelle instance pointant sur le même dossier
    store2 = ChromaStore(cfg)
    assert store2.count() == 1
    fetched = store2.get_chunk(sample_chunk_structural_1.chunk_id)
    assert fetched["chunk_id"] == sample_chunk_structural_1.chunk_id
    assert fetched["metadata"]["sensitivity_level"] == 3

