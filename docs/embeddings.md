# Architecture des Embeddings et Stockage Vectoriel ChromaDB — DataNova Technologies

Ce document décrit l'architecture, la modélisation et la stratégie d'indexation vectorielle développées lors de la **Phase 6** du projet RAG Multi-Sources de DataNova Technologies.

---

## 1. Vue d'ensemble du pipeline

Le module d'embedding et d'indexation vectorielle transforme les fragments de texte produits en Phase 5 (`data/processed/chunks_structural.json` et `chunks_fixed.json`) en vecteurs denses et les stocke de façon reproductible et persistante dans ChromaDB.

```
Chunks normalisés (Phase 5)
  [chunks_structural.json (142 chunks)]
               │
               ▼
┌──────────────────────────────┐
│     TokenizerInspector       │  -> Audit exhaustif des tokens (507 max)
│  (Anti-troncature stricte)   │     Zéro troncature silencieuse
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      EmbeddingManager        │  -> Modèle: intfloat/multilingual-e5-small
│  (multilingual-e5-small)     │  -> Injection en mémoire de 'passage: '
└──────────────┬───────────────┘  -> Batching CPU (batch_size = 16)
               │
               ▼
┌──────────────────────────────┐
│     EmbeddingValidator       │  -> Contrôle dimension (384), finitude (NaN/Inf)
│   (Contrôle d'intégrité)     │     Vecteurs non nuls
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│        ChromaStore           │  -> Base persistante unique (vector_store/)
│ (Reconstruction Blue/Green)  │  -> 2 collections étanches
└──────────────┬───────────────┘  -> Omission des None, types primitifs stricts
               │
     ┌─────────┴─────────┐
     ▼                   ▼
Collection            Collection
Structurelle          Fixe
(142 chunks)          (195 chunks)
```

---

## 2. Modèle d'Embedding : `intfloat/multilingual-e5-small`

### 2.1 Justification du choix
1. **Longueur de séquence de 512 tokens :** Permet d'englober sans aucune troncature nos sections Markdown les plus longues (~1 146 caractères, ~280 tokens).
2. **Dimension de 384 :** Représentation vectorielle compacte (~1,5 Ko par chunk) limitant l'espace disque et la consommation de RAM.
3. **Révision immuable :** Figeage strict sur le commit SHA-1 `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
4. **Préfixes asymétriques E5 :**
   - À l'indexation : préfixe `"passage: "` injecté **exclusivement en mémoire tensorielle** (le texte stocké dans ChromaDB reste 100 % pur et intact).
   - À la recherche (Phase 7) : préfixe `"query: "` pour les requêtes utilisateur.

---

## 3. Procédure de Reconstruction Sécurisée (*Blue/Green Staging*)

Afin de pallier l'absence d'opération atomique de renommage dans ChromaDB et d'éviter la suppression prématurée de l'index actif :

1. Création d'une collection temporaire isolée : `datanova_corpus_structural__staging`.
2. Insertion et validation complète des embeddings et métadonnées dans la collection Staging.
3. Vérification de la cardinalité ($142 / 142$).
4. Si la validation réussit : suppression de l'ancienne collection active et transfert direct depuis Staging.
5. Suppression de la collection Staging après succès.
6. En cas d'incident avant validation : Staging est supprimé, l'ancienne collection active reste **100 % intacte et opérationnelle**.

---

## 4. Spécification des Métadonnées et Contrôle RBAC

ChromaDB exige des types primitifs (`str`, `int`, `float`, `bool`).

### 4.1 Métadonnées universelles (Présentes sur 100 % des chunks)
- `document_id` : Identifiant parent (ex: `DOC-SEC-001`).
- `source` : Chemin source normalisé.
- `date_version` : Version documentaire.
- `doc_type` : Format (`pdf`, `markdown`, `faq`, `csv`).
- `sensitivity_level` : Niveau RBAC ($1, 2, 3$).
- `chunk_id` : Identifiant déterministe unique.
- `chunk_index` : Index séquentiel ($0 \le i < \text{total\_chunks}$).
- `total_chunks` : Total de chunks du document parent.
- `chunking_strategy` : `"structural"` ou `"fixed"`.

### 4.2 Métadonnées spécifiques
- PDF : `page_number`, `total_pages`, `title`.
- FAQ : `faq_id`, `category`, `title`.
- CSV : `dataset_id`, `domain`, `storage_layer`, `storage_format`, `update_frequency`, `retention_period`, `access_policy`.
- Markdown : `section`, `title`.

### 4.3 Règle d'omission des valeurs nulles
Toute métadonnée avec `val is None` est omise. Aucune fausse chaîne (`"None"`, `"null"`) n'est injectée artificiellement.

