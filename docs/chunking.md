# Architecture et Stratégies de Chunking Documentaire — DataNova Technologies

Ce document décrit l'architecture, la modélisation et les stratégies de découpage documentaire développées lors de la **Phase 5** du projet RAG Multi-Sources de DataNova Technologies.

---

## 1. Vue d'ensemble

Le module de chunking récupère les **66 documents normalisés** issus du pipeline d'ingestion (Phase 4) en mémoire, et applique deux stratégies distinctes de découpage en fragments prêts pour l'indexation vectorielle (Phase 6).

```
Documents normalisés (Phase 4)
  [66 NormalizedDocument en mémoire]
               │
               ▼
   ┌───────────────────────┐
   │    ChunkingManager    │
   └───────────┬───────────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
┌──────────────┐   ┌─────────────────┐
│ Stratégie A  │   │   Stratégie B   │
│ Découpage    │   │   Découpage     │
│ Fixe         │   │   Structurel    │
└──────┬───────┘   └────────┬────────┘
       │                    │
       ▼                    ▼
┌────────────────────────────────────┐
│          ChunkValidator            │
│ (Intégrité, RBAC, ChromaDB, Unicité)│
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│      Stockage Reproductible        │
│ • data/processed/chunks_fixed.json │
│ • data/processed/chunks_struct.json│
└────────────────────────────────────┘
```

---

## 2. Modèle de Données `ChunkDocument`

Chaque chunk est représenté par la structure Pydantic [`ChunkDocument`](file:///c:/Users/Salma/Desktop/RAG-multi-sources/src/chunking/models.py) :

```python
class ChunkDocument(BaseModel):
    chunk_id: str
    content: str
    metadata: Dict[str, Union[str, int, float, bool]]
```

### 2.1 Socle universel de métadonnées
Chaque chunk conserve strictement les 5 métadonnées obligatoires du document d'origine :
1. `document_id` : Identifiant du fichier source parent dans `inventory.json`.
2. `source` : Chemin relatif normalisé (ex: `data/raw/pdf/...`).
3. `date_version` : Version du document ou date de mise à jour.
4. `doc_type` : Type documentaire (`pdf`, `markdown`, `faq`, `csv`).
5. `sensitivity_level` : Niveau d'accréditation RBAC (entier $\in \{1, 2, 3\}$).

### 2.2 Métadonnées d'orchestration
- `chunk_id` : Identifiant déterministe unique.
- `chunk_index` : Index séquentiel du chunk au sein du document parent ($0 \le i < \text{total\_chunks}$).
- `total_chunks` : Nombre total de chunks produits pour ce document parent.
- `chunking_strategy` : `"fixed"` ou `"structural"`.

### 2.3 Métadonnées spécifiques préservées
- **PDF :** `page_number`, `total_pages`.
- **FAQ :** `faq_id`, `category`.
- **CSV :** `dataset_id`, `domain`, `owner_team`, `storage_layer`, etc.
- **Markdown :** `section` (titre de la section parente).

---

## 3. Identifiants Déterministes et Unicité (`chunk_id`)

Pour garantir une **reproductibilité binaire parfaite** sans dépendre d'un timestamp ou d'un identifiant aléatoire (UUID), la formule de génération est :

$$\text{chunk\_id} = \text{parent\_doc\_id} + \text{"\_"} + \text{strategy\_prefix} + \text{"\_c"} + \text{chunk\_index} + \text{"\_"} + \text{SHA256(content)[:12]}$$

Exemples :
- `DOC-SEC-001#p1_str_c0_7a8b9c0d1e2f`
- `FAQ-DE-001_fix_c0_9a1f2b3c4d5e`
- `DS-001_str_c0_4d5e6f7a8b9c`

---

## 4. Comparaison des Deux Stratégies

### Configuration A — Chunking fixe (`FixedTextSplitter`)
- Découpage uniforme via `RecursiveCharacterTextSplitter`.
- Taille indicative : `chunk_size = 500` caractères, `chunk_overlap = 50` caractères.
- Séparateurs : `["\n\n", "\n", ". ", " ", ""]`.

### Configuration B — Chunking structurel (`StructuralTextSplitter`)
- **FAQ (`faq`) :** Maintien de l'intégrité de la paire Question / Réponse dans 1 chunk unitaire.
- **CSV (`csv`) :** Maintien de la fiche dataset complète (12 attributs) dans 1 chunk unitaire.
- **Markdown (`markdown`) :** Découpage guidé par les titres majeurs (`#` et `##`). Les sections restent unitaires jusqu'à un seuil de 1 200 caractères ; seul le dépassement entraîne un sous-découpage récursif.
- **PDF (`pdf`) :** Respect strict des limites de page (pas de fusion inter-page) et découpage intra-page par paragraphes (`chunk_size = 600`, `chunk_overlap = 60`).

---

## 5. Validateur Qualité (`ChunkValidator`)

Le validateur vérifie :
- L'absence de contenu vide.
- La présence intégrale des métadonnées obligatoires.
- La conformité du niveau de sensibilité ($\in \{1, 2, 3\}$).
- L'absence d'objets complexes (compatibilité ChromaDB).
- L'unicité absolue des `chunk_id` dans le lot (zéro collision).
- La cohérence de l'index séquentiel ($0 \le \text{chunk\_index} < \text{total\_chunks}$).
- Les avertissements informatifs (chunks très courts ou dépassant la taille indicative sans rejet bloquant).

