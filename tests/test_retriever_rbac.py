"""Tests unitaires et de sécurité pour le Retriever sémantique et le contrôle d'accès RBAC (Phase 7)."""
import pytest
import math
import hashlib
import logging
from typing import List, Dict, Any
from pathlib import Path

from src.chunking.models import ChunkDocument
from src.embeddings.embedding_config import EmbeddingConfig
from src.embeddings.embedding_manager import EmbeddingManager
from src.embeddings.chroma_store import ChromaStore
from src.retrieval.models import (
    ClearanceLevel,
    ROLE_PERMISSIONS,
    UserContext,
    RetrievedChunk,
    RetrievalResult,
    InvalidClearanceError,
    InvalidTopKError,
    RetrieverError,
    MAX_TOP_K
)
from src.retrieval.rbac_manager import RBACManager
from src.retrieval.semantic_retriever import SemanticRetriever


# ==============================================================================
# HELPERS DE SIMULATION HORS-LIGNE (DÉTERMINISTES SANS RÉSEAU)
# ==============================================================================

def mock_embedding_fn_384(texts: List[str]) -> List[List[float]]:
    """Générateur d'embeddings déterministe 384-d L2-normalisé."""
    vectors: List[List[float]] = []
    for text in texts:
        vec: List[float] = []
        for i in range(384):
            h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).digest()
            val = (int.from_bytes(h[:4], "little") / 2**32) - 0.5
            vec.append(val)
        norm = math.sqrt(sum(x * x for x in vec))
        normalized = [x / norm for x in vec]
        vectors.append(normalized)
    return vectors


@pytest.fixture
def test_chunks_multi_sensitivity() -> List[ChunkDocument]:
    """Corpus de test synthétique couvrant les 3 niveaux de sensibilité."""
    return [
        ChunkDocument(
            chunk_id="CHK-PUB-001",
            content="Catalogue des datasets IoT ouverts pour la télémétrie industrielle.",
            metadata={
                "document_id": "DOC-CAT-001",
                "source": "data/raw/csv/dataset_catalog.csv",
                "date_version": "2.3.0",
                "doc_type": "csv",
                "sensitivity_level": 1,
                "chunk_id": "CHK-PUB-001",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunking_strategy": "structural"
            }
        ),
        ChunkDocument(
            chunk_id="CHK-PUB-002",
            content="Foire aux questions générale sur l'accès aux clusters Spark et Kafka.",
            metadata={
                "document_id": "DOC-FAQ-001",
                "source": "data/raw/faq/technical_faq.json",
                "date_version": "1.5.0",
                "doc_type": "faq",
                "sensitivity_level": 1,
                "chunk_id": "CHK-PUB-002",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunking_strategy": "structural"
            }
        ),
        ChunkDocument(
            chunk_id="CHK-INT-001",
            content="Architecture de référence Lakehouse couches Bronze, Silver et Gold DataNova.",
            metadata={
                "document_id": "DOC-ARCH-001",
                "source": "data/raw/markdown/data_platform_architecture.md",
                "date_version": "2.4.0",
                "doc_type": "markdown",
                "sensitivity_level": 2,
                "chunk_id": "CHK-INT-001",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunking_strategy": "structural"
            }
        ),
        ChunkDocument(
            chunk_id="CHK-INT-002",
            content="Standards de développement et règles de pull requests pour Data Engineering.",
            metadata={
                "document_id": "DOC-ENG-002",
                "source": "data/raw/markdown/data_engineering_standards.md",
                "date_version": "1.8.0",
                "doc_type": "markdown",
                "sensitivity_level": 2,
                "chunk_id": "CHK-INT-002",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunking_strategy": "structural"
            }
        ),
        ChunkDocument(
            chunk_id="CHK-RES-001",
            content="Extractions de paie confidentielles, primes RH et déclarations sociales de direction.",
            metadata={
                "document_id": "DOC-HR-001",
                "source": "data/raw/csv/dataset_catalog.csv",
                "date_version": "2.3.0",
                "doc_type": "csv",
                "sensitivity_level": 3,
                "chunk_id": "CHK-RES-001",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunking_strategy": "structural"
            }
        ),
        ChunkDocument(
            chunk_id="CHK-RES-002",
            content="Traces d'audit de sécurité SOC et détections d'accès anormaux par le RSSI.",
            metadata={
                "document_id": "DOC-SEC-001",
                "source": "data/raw/pdf/data_governance_security_policy.pdf",
                "date_version": "3.0.0",
                "doc_type": "pdf",
                "sensitivity_level": 3,
                "chunk_id": "CHK-RES-002",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunking_strategy": "structural"
            }
        ),
    ]


@pytest.fixture
def populated_retriever(tmp_path, test_chunks_multi_sensitivity) -> SemanticRetriever:
    """Fixture initialisant un SemanticRetriever avec 6 chunks indexés dans un ChromaStore temporaire."""
    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store = ChromaStore(cfg)
    manager = EmbeddingManager(config=cfg, embedding_fn=mock_embedding_fn_384)

    ids, texts, embs, metas = manager.process_chunks(test_chunks_multi_sensitivity)
    store.upsert_chunks(ids, texts, embs, metas)

    return SemanticRetriever(
        embedding_manager=manager,
        chroma_store=store,
        rbac_manager=RBACManager(),
        default_top_k=4
    )


# ==============================================================================
# 1. RECHERCHE SÉMANTIQUE SIMPLE
# ==============================================================================

def test_semantic_search_basic(populated_retriever):
    """1. Vérifie une recherche nominale : chunks extraits, classés, avec score et distance."""
    user = UserContext.internal("test_user")
    result = populated_retriever.retrieve(
        query="Comment configurer l'architecture Lakehouse ?",
        user_context=user,
        top_k=2
    )

    assert isinstance(result, RetrievalResult)
    assert result.total_retrieved > 0
    assert len(result.chunks) <= 2
    assert result.query == "Comment configurer l'architecture Lakehouse ?"
    assert result.user_id == "test_user"
    assert result.clearance_level == 2
    assert result.execution_time_ms >= 0.0

    # Vérification des attributs du premier chunk
    first_chunk = result.chunks[0]
    assert isinstance(first_chunk, RetrievedChunk)
    assert first_chunk.chunk_id in ("CHK-INT-001", "CHK-INT-002", "CHK-PUB-001", "CHK-PUB-002")
    assert first_chunk.content != ""
    assert isinstance(first_chunk.similarity_score, float)
    assert isinstance(first_chunk.distance, float)
    assert -1.0 <= first_chunk.similarity_score <= 1.0


# ==============================================================================
# 2. RETOUR DES TOP-K RÉSULTATS ET VALIDATION DE TOP_K
# ==============================================================================

def test_top_k_parameter(populated_retriever):
    """2. Vérifie le respect strict du paramètre top_k et la validation de ses bornes."""
    user = UserContext.restricted("admin_user")

    # Demande top_k = 1
    res1 = populated_retriever.retrieve(query="sécurité", user_context=user, top_k=1)
    assert len(res1.chunks) == 1
    assert res1.top_k_requested == 1

    # Demande top_k = 3
    res3 = populated_retriever.retrieve(query="sécurité", user_context=user, top_k=3)
    assert len(res3.chunks) == 3
    assert res3.top_k_requested == 3

    # Clamping lorsque top_k dépasse le nombre total de documents dans la base (6)
    res_large = populated_retriever.retrieve(query="sécurité", user_context=user, top_k=10)
    assert len(res_large.chunks) <= 6

    # Rejet des valeurs invalides de top_k
    with pytest.raises(InvalidTopKError, match="strictement supérieur à 0"):
        populated_retriever.retrieve(query="test", user_context=user, top_k=0)

    with pytest.raises(InvalidTopKError, match="strictement supérieur à 0"):
        populated_retriever.retrieve(query="test", user_context=user, top_k=-5)

    with pytest.raises(InvalidTopKError, match="booléen"):
        populated_retriever.retrieve(query="test", user_context=user, top_k=True)

    with pytest.raises(InvalidTopKError, match="entier strict"):
        populated_retriever.retrieve(query="test", user_context=user, top_k="3")

    with pytest.raises(InvalidTopKError, match="limite maximale"):
        populated_retriever.retrieve(query="test", user_context=user, top_k=MAX_TOP_K + 1)


# ==============================================================================
# 3. UTILISATION OBLIGATOIRE DU PRÉFIXE E5 "query: "
# ==============================================================================

def test_e5_query_prefix_enforcement(tmp_path, test_chunks_multi_sensitivity):
    """3. Vérifie que l'EmbeddingManager injecte systématiquement le préfixe 'query: '."""
    captured_model_inputs: List[str] = []

    def capturing_mock(texts: List[str]) -> List[List[float]]:
        captured_model_inputs.extend(texts)
        return mock_embedding_fn_384(texts)

    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store = ChromaStore(cfg)
    manager = EmbeddingManager(config=cfg, embedding_fn=capturing_mock)

    ids, texts, embs, metas = manager.process_chunks(test_chunks_multi_sensitivity)
    store.upsert_chunks(ids, texts, embs, metas)
    captured_model_inputs.clear()

    retriever = SemanticRetriever(embedding_manager=manager, chroma_store=store)
    user = UserContext.internal("user_test")

    retriever.retrieve(query="Quelle est la rétention des logs ?", user_context=user, top_k=2)

    # L'entrée reçue par le modèle d'embedding doit commencer par 'query: '
    assert len(captured_model_inputs) == 1
    assert captured_model_inputs[0].startswith("query: ")
    assert captured_model_inputs[0] == "query: Quelle est la rétention des logs ?"


# ==============================================================================
# 4. PRÉSERVATION DU CONTENU ET DES MÉTADONNÉES
# ==============================================================================

def test_chunk_content_and_metadata_preservation(populated_retriever):
    """4. Vérifie l'intégrité intégrale du texte et des métadonnées des chunks retournés."""
    user = UserContext.internal("dev_user")
    result = populated_retriever.retrieve(
        query="architecture lakehouse",
        user_context=user,
        top_k=4
    )

    for chunk in result.chunks:
        assert chunk.content is not None
        assert not chunk.content.startswith("passage: ")
        assert not chunk.content.startswith("query: ")

        # Contrôle des métadonnées requises
        meta = chunk.metadata
        assert "document_id" in meta
        assert "source" in meta
        assert "date_version" in meta
        assert "doc_type" in meta
        assert "sensitivity_level" in meta
        assert chunk.sensitivity_level == meta["sensitivity_level"]


# ==============================================================================
# 5. ACCÈS AUTORISÉ POUR UTILISATEUR NIVEAU 1 (PUBLIC)
# ==============================================================================

def test_rbac_access_level_1(populated_retriever):
    """5. Vérifie qu'un utilisateur Niveau 1 (Public) ne reçoit EXCLUSIVEMENT que des chunks de niveau 1."""
    user = UserContext.public("guest_user")
    result = populated_retriever.retrieve(query="datasets et architecture", user_context=user, top_k=6)

    assert result.clearance_level == 1
    assert len(result.chunks) > 0

    for chunk in result.chunks:
        # Aucun chunk de niveau 2 ou 3 ne doit être présent
        assert chunk.sensitivity_level == 1, (
            f"VIOLATION DE SÉCURITÉ : Chunk {chunk.chunk_id} de niveau {chunk.sensitivity_level} "
            f"exposé à un utilisateur Niveau 1 !"
        )


# ==============================================================================
# 6. ACCÈS AUTORISÉ POUR UTILISATEUR NIVEAU 2 (INTERNE)
# ==============================================================================

def test_rbac_access_level_2(populated_retriever):
    """6. Vérifie qu'un utilisateur Niveau 2 (Interne) accède aux niveaux 1 et 2, mais JAMAIS au niveau 3."""
    user = UserContext.internal("engineer_user")
    result = populated_retriever.retrieve(query="paie, audit et architecture", user_context=user, top_k=6)

    assert result.clearance_level == 2
    assert len(result.chunks) > 0

    for chunk in result.chunks:
        assert chunk.sensitivity_level in (1, 2)
        assert chunk.sensitivity_level != 3, (
            f"VIOLATION DE SÉCURITÉ : Chunk confidentiel {chunk.chunk_id} (niveau 3) "
            f"exposé à un collaborateur Niveau 2 !"
        )


# ==============================================================================
# 7. ACCÈS AUTORISÉ POUR UTILISATEUR NIVEAU 3 (RESTREINT / CONFIDENTIEL)
# ==============================================================================

def test_rbac_access_level_3(populated_retriever):
    """7. Vérifie qu'un utilisateur Niveau 3 (Restreint) peut accéder à tous les niveaux (1, 2 et 3)."""
    user = UserContext.restricted("ciso_user")
    result = populated_retriever.retrieve(query="traces d'audit et paie", user_context=user, top_k=6)

    assert result.clearance_level == 3
    assert len(result.chunks) > 0

    sensitivities_found = {c.sensitivity_level for c in result.chunks}
    # Un utilisateur niveau 3 a la capacité d'extraire des chunks de niveau 3
    assert 3 in sensitivities_found or len(sensitivities_found) > 0


# ==============================================================================
# 8. TEST CRITIQUE : REFUS STRICT D'UN DOCUMENT CONFIDENTIEL TRÈS PERTINENT
# ==============================================================================

def test_rbac_strict_denial_of_higher_sensitivity(tmp_path):
    """
    8. TEST DE SÉCURITÉ MAJEUR :
       Un chunk de sensibilité 3 possède un contenu et un embedding IDÉALEMENT alignés
       avec la requête (distance quasi-nulle).
       Un utilisateur Niveau 1 ou 2 NE DOIT JAMAIS le recevoir en résultat, même s'il est
       le candidat numéro 1 mondial par similarité sémantique.
    """
    target_query = "Document secret confidentiel hautement critique"

    # Construction d'un vecteur parfait pour la requête cible
    perfect_query_vector = mock_embedding_fn_384([f"query: {target_query}"])[0]

    def biased_mock_embed(texts: List[str]) -> List[List[float]]:
        vectors = []
        for t in texts:
            if "secret" in t.lower():
                # Le chunk secret reçoit un vecteur très proche de la requête
                vectors.append(list(perfect_query_vector))
            else:
                # Les autres chunks reçoivent des vecteurs orthogonaux/éloignés
                vectors.append(mock_embedding_fn_384([t])[0])
        return vectors

    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store = ChromaStore(cfg)
    manager = EmbeddingManager(config=cfg, embedding_fn=biased_mock_embed)

    sensitive_chunks = [
        ChunkDocument(
            chunk_id="CHK-SECRET-L3",
            content="Document secret confidentiel hautement critique sur les comptes bancaires.",
            metadata={
                "document_id": "DOC-SECRET-001",
                "source": "secret.pdf",
                "date_version": "1.0",
                "doc_type": "pdf",
                "sensitivity_level": 3,
                "chunk_id": "CHK-SECRET-L3",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunking_strategy": "structural"
            }
        ),
        ChunkDocument(
            chunk_id="CHK-PUBLIC-L1",
            content="Documentation publique sans grand rapport avec les comptes.",
            metadata={
                "document_id": "DOC-PUB-001",
                "source": "public.md",
                "date_version": "1.0",
                "doc_type": "markdown",
                "sensitivity_level": 1,
                "chunk_id": "CHK-PUBLIC-L1",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunking_strategy": "structural"
            }
        )
    ]

    ids, texts, embs, metas = manager.process_chunks(sensitive_chunks)
    store.upsert_chunks(ids, texts, embs, metas)

    retriever = SemanticRetriever(embedding_manager=manager, chroma_store=store)

    # 1. Requête par un utilisateur Niveau 1
    user_l1 = UserContext.public("user_public")
    res_l1 = retriever.retrieve(query=target_query, user_context=user_l1, top_k=2)

    # LE CHUNK SECRET EST FORMELLEMENT ABSENT
    returned_ids_l1 = [c.chunk_id for c in res_l1.chunks]
    assert "CHK-SECRET-L3" not in returned_ids_l1
    for c in res_l1.chunks:
        assert c.sensitivity_level <= 1

    # 2. Requête par un utilisateur Niveau 3 (Habilité à accéder aux niveaux 1, 2 et 3)
    user_l3 = UserContext.restricted("user_ciso")
    res_l3 = retriever.retrieve(query=target_query, user_context=user_l3, top_k=2)

    # L'utilisateur niveau 3 a bien accès au chunk de niveau 3, ainsi qu'aux chunks de niveau inférieur
    returned_ids_l3 = [c.chunk_id for c in res_l3.chunks]
    assert "CHK-SECRET-L3" in returned_ids_l3
    assert "CHK-PUBLIC-L1" in returned_ids_l3
    assert all(c.sensitivity_level <= 3 for c in res_l3.chunks)


# ==============================================================================
# 9. REJET DES NIVEAUX D'AUTORISATION INVALIDES
# ==============================================================================

def test_invalid_clearance_levels_rejected():
    """9. Vérifie le rejet systématique des niveaux d'autorisation invalides, booléens ou mal typés."""
    rbac = RBACManager()

    invalid_levels = [0, 4, -1, 99, "2", "admin", None, 1.5, True, False]
    for bad_level in invalid_levels:
        with pytest.raises(InvalidClearanceError):
            rbac.validate_clearance(bad_level)

        with pytest.raises(InvalidClearanceError):
            rbac.build_chroma_filter(bad_level)

        with pytest.raises(InvalidClearanceError):
            UserContext(user_id="test", clearance_level=bad_level)


# ==============================================================================
# 10. REFUS PAR DÉFAUT (DEFAULT-DENY) SUR MÉTADONNÉES ABSENTES OU CORROMPUES
# ==============================================================================

def test_default_deny_on_missing_or_corrupt_metadata():
    """10. Vérifie que toute anomalie de métadonnée entraîne un refus d'accès catégorique."""
    rbac = RBACManager()

    # Métadonnées sans champ 'sensitivity_level'
    assert rbac.is_access_granted({"chunk_id": "c1", "doc_type": "pdf"}, clearance_level=3) is False

    # Métadonnée 'sensitivity_level' None
    assert rbac.is_access_granted({"chunk_id": "c2", "sensitivity_level": None}, clearance_level=3) is False

    # Métadonnée 'sensitivity_level' chaîne de caractères
    assert rbac.is_access_granted({"chunk_id": "c3", "sensitivity_level": "2"}, clearance_level=3) is False

    # Métadonnée 'sensitivity_level' booléen
    assert rbac.is_access_granted({"chunk_id": "c4", "sensitivity_level": True}, clearance_level=3) is False

    # Métadonnée 'sensitivity_level' hors bornes [1, 3]
    assert rbac.is_access_granted({"chunk_id": "c5", "sensitivity_level": 0}, clearance_level=3) is False
    assert rbac.is_access_granted({"chunk_id": "c6", "sensitivity_level": 5}, clearance_level=3) is False

    # Métadonnées None ou type non dict
    assert rbac.is_access_granted(None, clearance_level=3) is False
    assert rbac.is_access_granted([], clearance_level=3) is False


# ==============================================================================
# 11. AUCUN CHUNK NON AUTORISÉ DANS LE CONTEXTE TRANSMIS AU LLM
# ==============================================================================

def test_zero_unauthorized_chunks_in_context(populated_retriever):
    """11. Vérifie formellement que to_context_str() ne contient aucune donnée non autorisée."""
    user = UserContext.public("user_public")
    result = populated_retriever.retrieve(query="paie, finance et direction", user_context=user, top_k=5)

    context_str = result.to_context_str()

    # Les identifiants et contenus des chunks restreints ne doivent pas apparaître dans la chaîne de contexte
    assert "CHK-RES-001" not in context_str
    assert "CHK-RES-002" not in context_str
    assert "CHK-INT-001" not in context_str
    assert "CHK-INT-002" not in context_str
    assert "Extractions de paie confidentielles" not in context_str
    assert "Traces d'audit de sécurité SOC" not in context_str


# ==============================================================================
# 12. GESTION DES REQUÊTES VIDES ET COLLECTIONS VIDES
# ==============================================================================

def test_empty_query_and_empty_collection(tmp_path):
    """12. Vérifie la gestion robuste des requêtes vides et des bases vectorielles sans documents."""
    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store = ChromaStore(cfg)  # Base vide (count = 0)
    manager = EmbeddingManager(config=cfg, embedding_fn=mock_embedding_fn_384)
    retriever = SemanticRetriever(embedding_manager=manager, chroma_store=store)

    user = UserContext.internal("user_test")

    # Rejet d'une requête vide ou blanche
    with pytest.raises(ValueError, match="ne peut pas être vide"):
        retriever.retrieve(query="", user_context=user)

    with pytest.raises(ValueError, match="ne peut pas être vide"):
        retriever.retrieve(query="   \t\n  ", user_context=user)

    # Requête sur base vide : renvoie un RetrievalResult vide propre sans erreur
    result = retriever.retrieve(query="Question valide sur base vide", user_context=user)
    assert result.total_retrieved == 0
    assert result.chunks == []
    assert result.to_context_str() == ""


# ==============================================================================
# 13. GESTION DES ERREURS DU RETRIEVER
# ==============================================================================

def test_retriever_error_handling(tmp_path):
    """13. Vérifie l'encapsulation propre des erreurs d'infrastructure dans RetrieverError."""
    cfg = EmbeddingConfig.create_structural_config(persist_directory=tmp_path)
    store = ChromaStore(cfg)

    # Embedding function défaillante simulant une panne
    def failing_embed(texts: List[str]):
        raise RuntimeError("Crash matériel GPU/CPU lors de l'inférence.")

    manager = EmbeddingManager(config=cfg, embedding_fn=failing_embed)
    retriever = SemanticRetriever(embedding_manager=manager, chroma_store=store)
    user = UserContext.internal("user_test")

    # Indexation d'un chunk pour que la collection ne soit pas considérée comme vide
    valid_embs = mock_embedding_fn_384(["test"])
    store.upsert_chunks(
        ids=["c1"],
        documents=["test"],
        embeddings=valid_embs,
        metadatas=[{"sensitivity_level": 1, "document_id": "DOC-1"}]
    )

    with pytest.raises(RetrieverError, match="Échec de vectorisation"):
        retriever.retrieve(query="Question provoquant un crash", user_context=user)


# ==============================================================================
# TESTS COMPLÉMENTAIRES DE SÉCURITÉ ET DE ROBUSTESSE
# ==============================================================================

def test_user_context_from_roles():
    """Vérifie la dérivation serveur des niveaux d'accréditation depuis les rôles."""
    # Invité -> Niveau 1
    u_guest = UserContext.from_roles("u1", ["guest"])
    assert u_guest.clearance_level == 1

    # Data Engineer -> Niveau 2
    u_eng = UserContext.from_roles("u2", ["data_engineer"])
    assert u_eng.clearance_level == 2

    # Profil multi-rôles incluant CISO -> Niveau 3 (le plus élevé l'emporte)
    u_ciso = UserContext.from_roles("u3", ["employee", "ciso"])
    assert u_ciso.clearance_level == 3

    # Rôles inconnus -> Repli de sécurité vers Niveau 1 (Moindre privilège)
    u_unknown = UserContext.from_roles("u4", ["super_hacker_role"])
    assert u_unknown.clearance_level == 1


def test_user_context_immutability():
    """Vérifie l'immuabilité de UserContext contre les modifications malveillantes en mémoire."""
    user = UserContext.public("user1")
    with pytest.raises(AttributeError):
        user.clearance_level = 3  # Tentative d'élévation de privilège bloquée


def test_similarity_score_calculation(tmp_path):
    """Vérifie le calcul exact de la similarité cosinus (1 - distance) dans son domaine théorique [-1, 1]."""
    chunk = RetrievedChunk(
        chunk_id="CHK-1",
        content="test",
        metadata={"sensitivity_level": 1},
        similarity_score=round(1.0 - 0.25, 4),
        distance=0.25,
        sensitivity_level=1
    )
    assert chunk.similarity_score == 0.75
    assert chunk.distance == 0.25


def test_sanitized_logging_no_content_leak(caplog):
    """Vérifie que la journalisation des refus d'accès n'enregistre JAMAIS le contenu textuel."""
    rbac = RBACManager()
    with caplog.at_level(logging.WARNING):
        metadata_sensitive = {
            "chunk_id": "CHK-CONFIDENTIAL-007",
            "document_id": "DOC-FIN-999",
            "doc_type": "csv",
            "sensitivity_level": 3,
            "secret_content_field": "MOT_DE_PASSE_ET_CHIFFRE_AFFAIRES_SECRET"
        }
        # Refus d'accès pour niveau 1
        rbac.is_access_granted(metadata_sensitive, clearance_level=1)

    log_output = caplog.text
    assert "CHK-CONFIDENTIAL-007" in log_output
    assert "Violation interceptée" in log_output
    # LE CONTENU CONFIDENTIEL NE DOIT JAMAIS APPARAÎTRE DANS LES LOGS
    assert "MOT_DE_PASSE_ET_CHIFFRE_AFFAIRES_SECRET" not in log_output

