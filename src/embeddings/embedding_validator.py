"""Validateur de qualité, d'intégrité numérique et de dimension pour les vecteurs d'embedding."""
import math
import logging
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class EmbeddingValidator:
    """
    Validateur rigoureux des vecteurs numériques avant stockage dans ChromaDB.
    
    Contrôles d'intégrité :
      1. Dimension exacte : le vecteur doit comporter exactement `expected_dim` (384) dimensions.
      2. Finitude numérique : aucun élément ne doit être NaN (Not a Number) ou infini (Inf).
      3. Vecteur non nul : le vecteur ne doit pas être un vecteur trivial (norme strictement positive).
      4. Alignement bijectif : dans un batch, le nombre de vecteurs doit correspondre exactement au nombre de chunks.
    """

    def __init__(self, expected_dim: int = 384):
        self.expected_dim = expected_dim

    def validate_vector(self, vector: List[float]) -> Tuple[bool, Optional[str]]:
        """
        Valide un vecteur individuel.
        Retourne (is_valid, reject_reason).
        """
        if not isinstance(vector, (list, tuple)) and not hasattr(vector, "__len__"):
            return False, f"Type de vecteur invalide : '{type(vector).__name__}' (attendu: liste ou tableau numérique)."

        # 1. Vérification dimensionnelle
        dim = len(vector)
        if dim != self.expected_dim:
            return False, f"Dimension incorrecte : {dim} dimensions observées (attendu: {self.expected_dim})."

        # 2. Vérification mathématique (NaN / Inf) et non-nullité
        has_non_zero = False
        for idx, val in enumerate(vector):
            try:
                f_val = float(val)
            except (ValueError, TypeError):
                return False, f"Composante non numérique à l'index {idx} : '{val}'."

            if math.isnan(f_val):
                return False, f"Valeur NaN détectée à l'index {idx} du vecteur."
            if math.isinf(f_val):
                return False, f"Valeur infinie (Inf) détectée à l'index {idx} du vecteur."
            if f_val != 0.0:
                has_non_zero = True

        # 3. Vérification vecteur non nul
        if not has_non_zero:
            return False, "Vecteur nul détecté (toutes les composantes sont égales à 0.0)."

        return True, None

    def validate_batch(
        self,
        vectors: List[List[float]],
        expected_count: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Valide un lot complet de vecteurs générés.
        Retourne (is_valid, reject_reason).
        """
        if len(vectors) != expected_count:
            return False, (
                f"Désalignement de cardinalité : {len(vectors)} vecteurs générés "
                f"pour {expected_count} chunks en entrée."
            )

        for idx, vec in enumerate(vectors):
            is_valid, reason = self.validate_vector(vec)
            if not is_valid:
                return False, f"Erreur sur le vecteur {idx} : {reason}"

        return True, None

