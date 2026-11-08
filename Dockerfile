# ==============================================================================
# Dockerfile - Système RAG Monolithique Sécurisé (RBAC)
# ==============================================================================
FROM python:3.11-slim

# Configuration des variables d'environnement Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Installation des dépendances système minimales (curl pour les healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Création d'un utilisateur non-root pour respecter les standards de sécurité
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

# Installation des dépendances Python (optimisation du cache Docker)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Préparation de l'arborescence
RUN mkdir -p /app/data /app/vector_store /app/mlruns /app/docs /app/src /app/tests && \
    chown -R appuser:appuser /app

# Copie du code source et des tests
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser tests/ /app/tests/

# Bascule vers l'utilisateur non-root
USER appuser

# Port exposé pour l'API RAG
EXPOSE 8000

# Healthcheck pour s'assurer de la disponibilité du service
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande de démarrage par défaut de l'application FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

