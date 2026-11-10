# Monitoring Minimal et Observabilité du RAG — DataNova Technologies

Ce document décrit le dispositif de **monitoring minimal, structuré et reproductible** mis en place lors de la **Phase 9** du projet RAG Multi-Sources, conformément aux sections 2.1, 2.2, 4, 8 et 9 du cahier des charges.

---

## 1. Vue d'ensemble et périmètre des dépendances

### 1.1 Zéro dépendance tierce pour le module de monitoring
Conformément aux exigences formulées par l'enseignant, le sous-système de monitoring (`src/monitoring/` et `scripts/analyze_logs.py`) est **strictement autonome** et repose **exclusivement sur la bibliothèque standard de Python** :
- `json` : sérialisation et désérialisation structurée JSON Lines (.jsonl).
- `time` : mesures de haute précision via `time.perf_counter()`.
- `uuid` : génération d'identifiants uniques UUIDv4 pour la corrélation.
- `re` : validation et assainissement strict des en-têtes de corrélation client.
- `threading` : exclusion mutuelle thread-safe (`threading.Lock`) pour l'écriture concurrente dans les journaux.
- `urllib.request` / `urllib.error` : sondes réseau légères pour interroger l'état du démon Ollama sans dépendance externe.
- `math` : calcul mathématique des percentiles par la méthode *Nearest-Rank*.
- `pathlib` : manipulation portable et sécurisée des chemins de fichiers.

> [!NOTE]
> **Périmètre des dépendances :** Le module de monitoring n'introduit aucun paquet externe (pas de Prometheus, Grafana, Alertmanager, Redis, Kafka ou `psutil`). Les bibliothèques tierces présentes dans le projet (`fastapi`, `uvicorn`, `requests`, `chromadb`, etc.) relèvent exclusivement du socle applicatif préexistant du serveur d'API et du pipeline RAG.

### 1.2 Principes directeurs
1. **Source unique de vérité pour la durée HTTP :** Le middleware ASGI (`observability_middleware` dans `src/api.py`) encadre l'intégralité du cycle de vie de la requête (`await call_next(request)`). Il constitue la source de référence unique pour la mesure de la durée totale HTTP et écrit **exactement un seul enregistrement par requête**.
2. **Validation et assainissement de `X-Request-ID` :** L'identifiant fourni par le client est accepté s'il vérifie l'expression `^[a-zA-Z0-9_.-]{1,64}$`. Tout identifiant vide, trop long (> 64 caractères) ou contenant des caractères invalides (injections de balises, retours à la ligne) est systématiquement remplacé par un UUIDv4 standard généré par le serveur.
3. **Cloisonnement strict et protection des données (Zero Leakage / RGPD) :**
   - **Aucun prompt complet** n'est écrit dans les logs.
   - **Aucune réponse textuelle générée** par le LLM n'est stockée.
   - **Aucun extrait de document ou chunk** n'est consigné.
   - **Aucun identifiant utilisateur direct (PII)** n'est utilisé comme étiquette de métrique. Seul le niveau d'accréditation résolu côté serveur (1, 2 ou 3) est tracé.

---

## 2. Indicateurs suivis et modes de calcul

| Indicateur | Définition technique | Mode de calcul / Formule | Provenance de la mesure |
| :--- | :--- | :--- | :--- |
| **Nombre total de requêtes** | Volume global des transactions traitées par l'API. | $N = \text{count}(\text{requêtes})$ | Middleware ASGI (`src/api.py`). |
| **Requêtes réussies / échouées** | Répartition selon la classe du code de statut HTTP. | Succès si $200 \le \text{status} < 400$, Échec si $\text{status} \ge 400$. | Middleware ASGI (`src/api.py`). |
| **Taux d'erreur HTTP** | Pourcentage de requêtes ayant abouti à une erreur client (4xx) ou serveur (5xx). | $\text{Taux} = \frac{\text{Requêtes en échec}}{N} \times 100$ | Journal structuré (`status_code`). |
| **Latence totale HTTP (Moyenne & P95)** | Temps de traitement de la transaction HTTP de bout en bout (incluant sérialisation). | Moyenne : $\frac{1}{N}\sum t_{\text{http}}$<br>P95 : méthode *Nearest-Rank* à 95 %. | Horodatage haute précision `time.perf_counter()` dans le middleware ASGI. |
| **Temps de Retrieval (Moyenne & P95)** | Durée d'interrogation et de préfiltrage vectoriel dans ChromaDB. | Moyenne et P95 calculés sur les requêtes ayant sollicité le retriever. | Horodatage dans `SemanticRetriever.retrieve()` (`execution_time_ms`). |
| **Temps de Génération LLM (Moyenne & P95)** | Durée d'inférence locale consommée par le modèle Ollama `llama3.2`. | Moyenne et P95 calculés **exclusivement sur les requêtes avec génération effective**. | Horodatage dans `OllamaClient.generate()` (`generation_time_ms`). |
| **Temps interne du Pipeline RAG** | Durée totale d'orchestration (Retrieval + Prompting + LLM). | $t_{\text{pipeline}} = t_{\text{retrieval}} + t_{\text{prompt}} + t_{\text{génération}}$ | Horodatage dans `RAGPipeline.run()` (`total_time_ms`). |
| **Disponibilité Ollama** | État opérationnel en direct du démon Ollama local. | Requête `GET /api/version` via `urllib.request` (timeout 2s). | Sonde standard dans `MetricsAnalyzer.check_ollama_live()`. |
| **Complétude documentaire** | Fréquence à laquelle le LLM a disposé d'éléments factuels suffisants. | Décompte des requêtes où `has_sufficient_context` vaut `True` vs `False`. | `RAGResponse.has_sufficient_context` via détection des marqueurs d'insuffisance. |
| **Répartition RBAC** | Volume de requêtes réparties par niveau de clearance certifié côté serveur. | Décompte par niveau résolu : 1 (Public), 2 (Interne), 3 (Restreint). | `user_context.clearance_level` résolu côté serveur via `SIMULATED_USER_REGISTRY`. |

---

## 3. Traitement des générations non exécutées

Lorsqu'une requête ne déclenche aucune inférence LLM (par exemple en **Situation A** où 0 document n'est autorisé pour l'utilisateur, ou en cas de repli dégradé suite à l'indisponibilité d'Ollama) :
- Le système consigne explicitement `"generation_time_ms": null` dans le journal JSONL (au lieu de `0.0`).
- Dans `metrics_analyzer.py`, les valeurs `null` / `None` sont **strictement exclues** du calcul de la moyenne et du P95 de génération.
- Cela évite de fausser artificiellement la latence moyenne d'inférence LLM à la baisse.
- Si le journal ne contient aucune génération effective (ou s'il est vide), l'analyseur indique explicitement :  
  `"Indisponible (aucune génération LLM effective enregistrée)"`.

---

## 4. Méthode mathématique du percentile P95

Pour évaluer la latence subie par les requêtes les plus lentes sans être faussé par des approximations, le calcul du **P95** applique la méthode mathématique standard du **rang le plus proche (*Nearest-Rank Method*)** :

1. L'ensemble des durées mesurées est trié par ordre croissant :
   $$S = [t_{(1)}, t_{(2)}, \dots, t_{(N)}]$$
2. Le rang $r$ correspondant au 95ᵉ percentile est calculé par la formule :
   $$r = \left\lceil \frac{95}{100} \times N \right\rceil$$
3. La valeur du P95 correspond au $r$-ième élément de la série triée :
   $$P_{95} = S[r]$$

*Exemple :* Sur une série de 10 requêtes, $r = \lceil 0{,}95 \times 10 \rceil = 10$. Le P95 est exactement la valeur de la 10ᵉ requête la plus lente.

---

## 5. Format du journal d'accès structuré

Les événements sont enregistrés ligne par ligne dans [`logs/api_access.jsonl`](file:///c:/Users/Salma/Desktop/RAG-multi-sources/logs/api_access.jsonl).

### Exemple d'enregistrement réel pour une requête RAG :
```json
{
  "timestamp": "2026-11-10T12:00:00.123456+00:00",
  "request_id": "8f3e2b1a-4c5d-4e6f-8a9b-0c1d2e3f4a5b",
  "endpoint": "/query",
  "method": "POST",
  "status_code": 200,
  "total_duration_ms": 1245.50,
  "retrieval_time_ms": 14.20,
  "generation_time_ms": 1210.30,
  "pipeline_time_ms": 1225.10,
  "has_sufficient_context": true,
  "resolved_clearance_level": 2,
  "error_type": null
}
```

### Exemple pour une requête sans génération LLM (Situation A / 0 chunk) :
```json
{
  "timestamp": "2026-11-10T12:01:00.654321+00:00",
  "request_id": "c4b1a8d2-9e3f-41a2-b8c7-5d6e7f8a9b0c",
  "endpoint": "/query",
  "method": "POST",
  "status_code": 200,
  "total_duration_ms": 16.80,
  "retrieval_time_ms": 14.10,
  "generation_time_ms": null,
  "pipeline_time_ms": 14.10,
  "has_sufficient_context": false,
  "resolved_clearance_level": 1,
  "error_type": null
}
```

---

## 6. Exécution du script d'analyse

### Commande PowerShell standard :
```powershell
.\.venv\Scripts\python.exe scripts/analyze_logs.py
```

*(Ou via l'invocation du module)* :
```powershell
.\.venv\Scripts\python.exe -m src.monitoring.metrics_analyzer
```

### Possibilité de spécifier un fichier personnalisé :
```powershell
.\.venv\Scripts\python.exe scripts/analyze_logs.py "chemin/vers/autre_fichier.jsonl"
```

---

## 7. Limites assumées du monitoring minimal

Conformément à la consigne de rigueur demandée par l'enseignant :
1. **Complétude du contexte vs Fidélité sémantique (Faithfulness) :**
   - L'indicateur `has_sufficient_context` détecte si le LLM a explicité un manque de données par ses marqueurs lexicaux habituels.
   - **Il ne remplace en aucun cas l'évaluation rigoureuse de la qualité** (*Faithfulness*, *Answer Relevance*, *Context Recall*) qui doit être menée sur un banc d'évaluation annoté avec vérité terrain (Ground Truth).
2. **Consommation des ressources système (CPU / RAM) :**
   - Conformément aux contraintes strictes d'un monitoring minimal sans dépendances tierces lourdes (telles que `psutil` ou agents hôtes), l'utilisation CPU et mémoire instantanée n'est pas instrumentée en direct dans ce module.
3. **Persistance des logs :**
   - Les journaux sont stockés localement dans le dossier `logs/`. Dans un environnement conteneurisé (Docker), ce dossier doit être monté en volume hôte persistant pour prévenir toute perte de données lors du renouvellement des conteneurs.
