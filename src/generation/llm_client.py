"""Client HTTP résilient pour l'inférence locale Ollama (Llama 3.2) avec support de simulation."""
import logging
import time
from typing import Optional, Tuple, Callable, Dict, Any
import httpx

from src.generation.models import (
    LLMConfig,
    LLMClientError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMModelNotFoundError
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client HTTP pour le serveur Ollama local.
    
    Garanties :
      - Utilise l'API REST native `/api/chat` avec options de décodage déterministes.
      - Gestion fine des erreurs (Connexion, Timeout, Modèle absent).
      - Support d'injection d'un faux générateur pour des tests unitaires 100% hors-ligne.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        generator_fn: Optional[Callable[[str, Optional[str]], str]] = None
    ):
        self.config = config or LLMConfig()
        self._generator_fn = generator_fn

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Génère une réponse textuelle à partir du prompt utilisateur et du prompt système.
        
        Retourne :
          (texte_généré, latence_en_millisecondes)
        """
        start_time = time.perf_counter()

        # 1. Mode simulation (Mock pour les tests sans serveur Ollama actif)
        if self._generator_fn is not None:
            try:
                answer = self._generator_fn(user_prompt, system_prompt)
            except Exception as e:
                raise LLMClientError(f"Erreur du générateur simulé : {e}")

            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return answer, latency_ms

        # 2. Construction de la requête pour Ollama /api/chat
        endpoint = f"{self.config.base_url.rstrip('/')}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "seed": self.config.seed,
                "num_predict": self.config.max_tokens,
                "num_ctx": getattr(self.config, "num_ctx", 4096)
            }
        }

        # 3. Exécution HTTP avec gestion d'erreurs typées
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.post(endpoint, json=payload)

                if response.status_code == 404:
                    raise LLMModelNotFoundError(
                        f"Le modèle '{self.config.model_name}' est introuvable sur le serveur Ollama "
                        f"({self.config.base_url}). Exécutez 'ollama pull {self.config.model_name}'."
                    )

                if response.status_code != 200:
                    raise LLMClientError(
                        f"Erreur HTTP {response.status_code} retournée par Ollama : {response.text}"
                    )

                data = response.json()
                answer = data.get("message", {}).get("content", "")

        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Impossible de joindre le serveur Ollama sur {self.config.base_url}. "
                f"Vérifiez que le service Ollama est démarré. Erreur : {e}"
            )
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"Délai d'attente dépassé ({self.config.timeout_seconds}s) lors de la génération avec Ollama. "
                f"Erreur : {e}"
            )
        except (LLMModelNotFoundError, LLMConnectionError, LLMTimeoutError, LLMClientError):
            raise
        except Exception as e:
            raise LLMClientError(f"Incident imprévu lors de l'appel à Ollama : {e}")

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return answer, latency_ms

    def check_health(self) -> bool:
        """Vérifie si le serveur Ollama est joignable."""
        if self._generator_fn is not None:
            return True

        endpoint = f"{self.config.base_url.rstrip('/')}/api/version"
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(endpoint)
                return res.status_code == 200
        except Exception:
            return False

