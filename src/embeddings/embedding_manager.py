"""Gestionnaire d'inférence pour la génération d'embeddings avec intfloat/multilingual-e5-small."""
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np

from src.chunking.models import ChunkDocument
from src.embeddings.embedding_config import EmbeddingConfig
from src.embeddings.embedding_validator import EmbeddingValidator

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Gestionnaire d'inférence vectorielle basé sur intfloat/multilingual-e5-small.
    
    Règles strictes :
      1. Injection systématique du préfixe 'passage: ' pour les documents/chunks lors de l'inférence.
      2. Le préfixe 'passage: ' est injecté UNIQUEMENT en mémoire tensorielle et JAMAIS dans le texte stocké.
      3. Inférence par lots (batch_size=16 par défaut) pour préserver la mémoire CPU.
      4. Support d'une fenêtre de 512 tokens sans troncature des sections longues du corpus.
      5. Validation dimensionnelle (384) et géométrique (absence de NaN/Inf) via EmbeddingValidator.
      6. Support d'injection d'un générateur d'embeddings simulé (mock) pour les tests unitaires hors ligne.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
    ):
        self.config = config or EmbeddingConfig.create_structural_config()
        self.validator = EmbeddingValidator(expected_dim=self.config.embedding_dim)
        self._model = None
        self._custom_embedding_fn = embedding_fn

    @property
    def model(self):
        """Chargement paresseux du modèle SentenceTransformer sur CPU avec son commit SHA-1 immuable."""
        if self._custom_embedding_fn is not None:
            return None

        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(
                f"Chargement du modèle d'embedding '{self.config.model_name}' "
                f"(commit: {self.config.model_revision[:12]}) sur {self.config.device}..."
            )
            self._model = SentenceTransformer(
                self.config.model_name,
                revision=self.config.model_revision,
                device=self.config.device
            )
            # Configuration explicite de la longueur de séquence maximale (512 tokens)
            if hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = self.config.max_seq_length

        return self._model

    def embed_query(self, query: str) -> List[float]:
        """
        Génère l'embedding d'une requête utilisateur en appliquant obligatoirement le préfixe 'query: '.
        Définit le contrat d'interface pour la Phase 7 (Retriever).
        """
        clean = query.strip() if query else ""
        if not clean:
            raise ValueError("Impossible de générer un embedding pour une requête vide.")

        return self.embed_text(clean, is_query=True)

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Génère l'embedding d'une chaîne unique en appliquant le bon préfixe (passage: ou query:)."""
        clean = text.strip() if text else ""
        if not clean:
            raise ValueError("Impossible de générer un embedding pour un texte vide.")

        vectors = self.embed_texts([clean], is_query=is_query)
        return vectors[0]

    def embed_texts(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Génère les embeddings pour une liste de textes par lots.
        Applique le préfixe 'query: ' si is_query=True, sinon 'passage: '.
        """
        if not texts:
            return []

        # Contrôle qu'aucun texte n'est vide
        for idx, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"Le texte à l'index {idx} est vide.")

        # Application du préfixe asymétrique E5 en mémoire tensorielle uniquement
        prefix = self.config.query_prefix if is_query else self.config.document_prefix
        prefixed_texts = [f"{prefix}{t.strip()}" for t in texts]

        all_vectors: List[List[float]] = []

        # Mode mocké pour les tests unitaires (rapide, déterministe et sans réseau)
        if self._custom_embedding_fn is not None:
            raw_vectors = self._custom_embedding_fn(prefixed_texts)
            for v in raw_vectors:
                all_vectors.append([float(x) for x in v])
        else:
            # Mode réel via SentenceTransformer
            batch_size = self.config.batch_size
            for i in range(0, len(prefixed_texts), batch_size):
                batch_texts = prefixed_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False
                )

                if isinstance(batch_embeddings, np.ndarray):
                    batch_list = batch_embeddings.tolist()
                else:
                    batch_list = [[float(val) for val in vec] for vec in batch_embeddings]

                all_vectors.extend(batch_list)

        # Validation stricte du lot complet généré
        is_valid, reason = self.validator.validate_batch(all_vectors, expected_count=len(texts))
        if not is_valid:
            raise ValueError(f"Échec de validation des vecteurs E5 : {reason}")

        return all_vectors

    def process_chunks(
        self,
        chunks: List[ChunkDocument]
    ) -> Tuple[List[str], List[str], List[List[float]], List[Dict[str, Any]]]:
        """
        Traite une liste de chunks pour l'indexation ChromaDB :
        Retourne : (ids, documents_text_propres, embeddings_calcules, metadatas_nettoyees).
        
        Important : documents_text conserve le texte original NON préfixé par 'passage: '.
        """
        if not chunks:
            return [], [], [], []

        ids: List[str] = [c.chunk_id for c in chunks]
        clean_texts: List[str] = [c.content for c in chunks]
        metadatas: List[Dict[str, Any]] = [dict(c.metadata) for c in chunks]

        # Calcul vectoriel avec injection du préfixe 'passage: ' en mémoire
        embeddings = self.embed_texts(clean_texts, is_query=False)

        return ids, clean_texts, embeddings, metadatas
