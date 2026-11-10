# Architecture de Génération RAG avec LLM Local — DataNova Technologies

Ce document détaille l'architecture, la configuration d'inférence locale, les mécanismes de prompt engineering sécurisé et les garanties d'ancrage factuel développés lors de la **Phase 8** du projet RAG Multi-Sources de DataNova Technologies.

---

## 1. Vue d'ensemble du pipeline de bout en bout

La Phase 8 parachève le système RAG monolithique en reliant le moteur de recherche sémantique avec contrôle d'accès RBAC (Phase 7) à un modèle de langage local exécuté via **Ollama** (`llama3.2`).

```
                    ┌──────────────────────────────┐
                    │     Requête Utilisateur      │
                    │   (Question + Rôles Session) │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │    UserContext (Backend)     │  -> Résolution serveur d'après rôles certifiés
                    │ (ClearanceLevel: 1, 2 ou 3)  │  -> Zéro confiance dans les entrées client
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  SemanticRetriever (Phase 7) │  -> Pre-filtering natif ChromaDB (where <= clearance)
                    │  (142 chunks vectorisés E5)  │  -> Post-validation applicative (Default-Deny)
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Contrôle de vacuité     │  -> Si 0 chunk autorisé : Réponse de repli propre
                    │    (Situation A vs B)        │     Le LLM N'EST PAS APPELÉ (économie de ressources)
                    └──────────────┬───────────────┘
                                   │ (Si chunks autorisés présents)
                                   ▼
                    ┌──────────────────────────────┐
                    │       PromptBuilder          │  -> Séparation hermétique système / utilisateur
                    │  (Défense Anti-Injection)    │  -> Encapsulation dans <untrusted_documents>
                    └──────────────┬───────────────┘  -> Échappement des balises XML fermantes
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      OllamaClient            │  -> Modèle local llama3.2 (HTTP /api/chat)
                    │ (llama3.2 / CPU 16 Go RAM)   │  -> Température 0.1, Seed 42, Max tokens 512
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       RAGResponse            │  -> Réponse textuelle étayée par [Document X]
                    │ (Citations + Latences MLOps) │  -> Liste 'sources' certifiée côté serveur
                    └──────────────────────────────┘
```

---

## 2. Choix et configuration du LLM local (Ollama)

### 2.1 Justification du modèle `llama3.2`
Pour un déploiement d'entreprise sur station de travail standard sous **Windows, 16 Go de RAM et sans GPU dédié** (inférence 100 % CPU) :

| Critère d'évaluation | Modèle `llama3.2` (3B Q4_K_M) | Modèles 7B+ (`mistral:7b`, `llama3.1:8b`) |
|---|---|---|
| **Poids sur disque** | ~2,0 Go | ~4,5 à 5,5 Go |
| **Empreinte RAM active** | **~3,5 à 4,2 Go** | ~7,0 à 9,5 Go |
| **Marge de RAM résiduelle** | **> 11 Go** (pour OS, embedding E5 et ChromaDB) | < 6 Go (risque de swap disque) |
| **Vitesse d'inférence CPU** | **15 à 25 tokens/seconde** | 3 à 6 tokens/seconde |
| **Latence moyenne (200 mots)** | **3 à 6 secondes** | 30 à 60 secondes |
| **Capacités multilingues** | Français et anglais natifs | Français et anglais natifs |
| **Fenêtre de contexte** | 128 000 tokens | 8 000 à 128 000 tokens |

Le modèle `llama3.2` (3 milliards de paramètres) offre le compromis optimal entre vitesse d'exécution sur CPU, respect rigoureux des consignes négatives du prompt système et empreinte mémoire contenue.

### 2.2 Paramètres de génération déterministes ([`LLMConfig`](file:///c:/Users/Salma/Desktop/RAG-multi-sources/src/generation/models.py#L36-L68))
Afin de minimiser le risque d'hallucination et de garantir la reproductibilité :
* `temperature = 0.1` : Distribution de probabilités resserrée sur les tokens les plus certains.
* `top_p = 0.9` : Échantillonnage de noyau limitant la dérive sémantique.
* `seed = 42` : Graine aléatoire fixée pour des réponses reproductibles à contexte identique.
* `max_tokens = 512` : Budget suffisant pour une réponse d'entreprise synthétique.
* `timeout_seconds = 60.0` : Délai de garde évitant le blocage indéfini des threads sur CPU.

---

## 3. Sécurité et Défense contre les Prompt Injections

Dans un pipeline RAG multi-sources, les documents ingérés (PDF, Markdown, etc.) constituent une surface d'attaque potentielle par **injection indirecte** (*indirect prompt injection*). Un fragment documentaire peut contenir des textes malveillants ordonnant au modèle d'ignorer ses règles ou de divulguer des secrets.

### 3.1 Priorité au contrôle applicatif externe
> [!IMPORTANT]
> Le prompt engineering constitue une couche de défense en profondeur, mais **jamais une garantie absolue**.
> La sécurité critique du système repose sur le fait que **le contrôle d'accès RBAC est opéré 100 % en dehors du LLM** :
> 1. Un utilisateur Niveau 1 ne peut JAMAIS avoir de document de Niveau 2 ou 3 injecté dans son contexte.
> 2. Le LLM ne dispose d'aucun outil, d'aucune fonction d'exécution de code (`function calling`), d'aucun accès terminal ni d'aucun pouvoir sur le système d'authentification.

### 3.2 Cloisonnement du Prompt Engineering
1. **Séparation des canaux système et utilisateur :**
   Les directives comportementales sont transmises via le rôle `system` de l'API Ollama, sanctuarisant les règles métier en tête de session.
2. **Encapsulation par balises XML fermées :**
   Les données extraites de la base sont confinées dans `<untrusted_documents> ... </untrusted_documents>`.
3. **Échappement des balises de délimitation ([`escape_xml_tags`](file:///c:/Users/Salma/Desktop/RAG-multi-sources/src/generation/prompt_builder.py#L42-L51)) :**
   Si un document contient la sous-chaîne `</untrusted_documents>`, elle est automatiquement neutralisée en `&lt;/untrusted_documents&gt;` afin d'empêcher toute fermeture anticipée du bloc de données.
4. **Consigne explicite de neutralisation :**
   Le prompt système ordonne formellement au modèle de considérer les textes documentaires comme des données passives et d'ignorer tout ordre qui y figurerait.

---

## 4. Gestion des Contextes Insuffisants

Le pipeline distingue rigoureusement deux situations :

### Situation A : Aucun chunk autorisé récupéré
* **Cause :** Question hors-sujet par rapport au corpus, ou ensemble des documents pertinents d'un niveau de sensibilité supérieur à l'accréditation de l'appelant.
* **Comportement du pipeline :**
  - **Le LLM N'EST PAS APPELÉ.** Aucune ressource CPU n'est consommée.
  - Réponse de repli immédiate : *"Les documents autorisés disponibles ne permettent pas de répondre à cette question (aucun document pertinent accessible pour votre niveau d'accréditation)."*
  - `sources = []`
  - `has_sufficient_context = False`
  - `generation_time_ms = 0.0`

### Situation B : Des chunks sont récupérés mais ne suffisent pas à répondre
* **Cause :** Les documents autorisés traitent du domaine mais ne contiennent pas la réponse à la question spécifique.
* **Comportement du pipeline :**
  - Le LLM est instruit d'analyser les faits présents, de signaler les éléments manquants et de refuser l'extrapolation.
  - Formule attendue : *"Les documents autorisés disponibles ne contiennent pas d'informations suffisantes pour répondre à cette question."*
  - Le pipeline inspecte la réponse via [`_evaluate_context_sufficiency`](file:///c:/Users/Salma/Desktop/RAG-multi-sources/src/generation/rag_pipeline.py#L47-L66) et positionne le flag `has_sufficient_context = False`.

---

## 5. Citations et Traçabilité de Provenance

### 5.1 Construction certifiée côté serveur
La liste des sources accompagnant la réponse ([`RAGResponse.sources`](file:///c:/Users/Salma/Desktop/RAG-multi-sources/src/generation/models.py#L144)) est **totalement découplée du texte généré par le LLM**. Elle est extraite directement des métadonnées des chunks retournés par le Retriever :

```python
@dataclass(frozen=True)
class DocumentCitation:
    document_id: str        # Ex: 'DOC-SEC-001'
    chunk_id: str           # Ex: 'DS-001_str_c0_19aa323fee27'
    source: str             # Ex: 'data/raw/csv/dataset_catalog.csv'
    doc_type: str           # Ex: 'csv', 'pdf', 'markdown', 'faq'
    sensitivity_level: int  # 1, 2 ou 3
    page_number: Optional[int] = None
    section: Optional[str] = None
    faq_id: Optional[str] = None
    dataset_id: Optional[str] = None
    reference: Optional[str] = None
    title: Optional[str] = None
```

### 5.2 Immunité contre les hallucinations de citations
Si le modèle de langage invente un faux repère documentaire dans sa réponse (ex : `[Document 999]`), ce marqueur textuel **n'est jamais admis dans la liste officielle des sources certifiées**. Seules les véritables sources extraites par le moteur ChromaDB constituent la provenance certifiée.

---

## 6. Intégration API FastAPI ([`src/api.py`](file:///c:/Users/Salma/Desktop/RAG-multi-sources/src/api.py))

L'API FastAPI expose la boucle RAG complète sur l'endpoint `POST /query` :

### 6.1 Résolution d'identité côté serveur
Pour pallier l'absence d'une infrastructure SSO externe tout en empêchant toute falsification client :
```python
# Registre certifié côté serveur (Mock IAM)
SIMULATED_USER_REGISTRY = {
    "alice": {"roles": ["data_engineer", "employee"]},  # Niveau 2
    "bob": {"roles": ["guest"]},                         # Niveau 1
    "charlie": {"roles": ["security_admin", "ciso"]}     # Niveau 3
}
```
L'accréditation est résolue côté backend. Tout paramètre de niveau arbitraire est strictement validé et borné par le moindre privilège.

### 6.2 Format de réponse de l'endpoint
```json
{
  "query": "Comment fonctionne le chiffrement des données ?",
  "answer": "D'après la politique de gouvernance [Document 1], le chiffrement est assuré par AES-256 avec gestion KMS...",
  "llm_model": "llama3.2",
  "sources_retrieved": [
    {
      "id": "DOC-SEC-001#p2_str_c0_...",
      "document_id": "DOC-SEC-001",
      "source": "data/raw/pdf/data_governance_security_policy.pdf",
      "doc_type": "pdf",
      "sensitivity_level": 2,
      "reference": "Page 2/3"
    }
  ],
  "retrieval_time_ms": 28.4,
  "generation_time_ms": 2450.8,
  "total_time_ms": 2479.2,
  "has_sufficient_context": true
}
```

---

## 7. Procédures d'Exécution et Commandes PowerShell

### 7.1 Lancement des tests automatisés de la Phase 8 (pytest)
```powershell
.\.venv\Scripts\pytest.exe tests/test_generation.py -v
```
*(Validation attendue : **13 tests passés avec succès** en moins de 2 secondes, grâce aux générateurs et embeddings simulés sans dépendance réseau).*

### 7.2 Validation globale de l'ensemble du projet (Phases 4 à 8)
```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```
*(Validation attendue : **56+ tests passés avec succès** sur l'ingestion, le chunking, les embeddings, le retrieval et la génération).*

### 7.3 Démonstration CLI du pipeline RAG complet
Assurez-vous qu'Ollama est démarré (`ollama serve`), puis exécutez :
```powershell
.\.venv\Scripts\python.exe -m src.generation.rag_pipeline
```

### 7.4 Lancement du serveur API FastAPI
```powershell
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000 --reload
```
Test de l'endpoint dans PowerShell :
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body '{"query": "Quels sont les datasets de télémétrie IoT ?", "user_id": "alice"}' | ConvertTo-Json -Depth 5
```

---

## 8. Limites Connues et Perspectives de Déploiement

1. **Simulation IAM en mémoire :** L'authentification des utilisateurs repose sur le dictionnaire interne `SIMULATED_USER_REGISTRY`. En environnement de production, ce composant sera substitué par une validation de token JWT (OAuth2 / OIDC) émis par l'Active Directory ou le fournisseur d'identité d'entreprise.
2. **Re-ranking sémantique avancé :** L'intégration future d'un cross-encoder de re-ranking (Phase MLOps ultérieure) permettra de réordonner finement les chunks avant injection dans le prompt.
3. **Streaming des réponses :** L'API opère actuellement en mode synchrone (`stream=False`). L'ajout du mode Server-Sent Events (SSE) pourra être envisagé pour afficher les tokens au fil de l'eau sur l'interface utilisateur.

