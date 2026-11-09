# Architecture du Retriever Sémantique et Contrôle d'Accès RBAC — DataNova Technologies

Ce document détaille l'architecture, le fonctionnement géométrique, le modèle de sécurité documentaire et les garanties d'isolation mis en œuvre lors de la **Phase 7** du projet RAG Multi-Sources de DataNova Technologies.

---

## 1. Vue d'ensemble de l'architecture

Le module de récupération documentaire (`src/retrieval/`) constitue le pont critique de sécurité entre l'index vectoriel persistant ChromaDB (Phase 6) et la future couche de génération par LLM (Phase 8).

Il a pour mission de sélectionner les fragments documentaires les plus pertinents en réponse à une requête utilisateur, tout en garantissant de façon stricte et étanche qu'**aucun fragment excédant le niveau d'autorisation de l'appelant ne puisse être retourné ni injecté dans le contexte du modèle génératif**.

```
Requête Utilisateur (texte brut)
            │
            ▼
┌──────────────────────────────┐
│       UserContext            │  -> Identité certifiée côté serveur (user_id, roles)
│  (Dérivation Backend RBAC)   │  -> clearance_level in {1, 2, 3} calculé côté backend
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      EmbeddingManager        │  -> Modèle: intfloat/multilingual-e5-small (384-d)
│   (Préfixe asymétrique)      │  -> Injection en mémoire de 'query: '
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Barrière 1 : Pre-filtering  │  -> Clause ChromaDB : where={'sensitivity_level': {'$lte': clearance}}
│       Natif ChromaDB         │  -> L'espace de résultats retournés est restreint aux autorisés
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Barrière 2 : Post-filtre    │  -> Contrôle unitaire applicatif de chaque chunk (Zero Trust)
│    applicatif Default-Deny   │  -> Rejet catégorique si métadonnée absente, corrompue ou > clearance
└──────────────┬───────────────┘  -> Journalisation d'audit sécurisée (sans fuite de texte)
               │
               ▼
┌──────────────────────────────┐
│      RetrievalResult         │  -> Contient EXCLUSIVEMENT des chunks autorisés
│ (Contexte prêt pour Phase 8) │  -> similarité = 1 - distance (domaine théorique [-1, 1])
└──────────────────────────────┘
```

---

## 2. Modèle de Sécurité RBAC et Gestion des Accréditations

### 2.1 Matrice des niveaux de sensibilité et d'accréditation
Le référentiel documentaire d'entreprise de DataNova Technologies classe les fragments selon 3 niveaux stricts :

| Niveau | Désignation | Description documentaire | Rôles associés (côté serveur) |
|---|---|---|---|
| **1** | **Public** | Catalogues ouverts de datasets (Bronze/Silver), FAQ générale, annuaire des collaborateurs. | `guest`, `anonymous`, `external_auditor`, `public` |
| **2** | **Interne** | Politiques de gouvernance PDF, architectures Lakehouse, standards de développement, procédures MLOps et Big Data. | `employee`, `data_engineer`, `data_scientist`, `mlops_engineer`, `devops_engineer` |
| **3** | **Restreint** | Extractions de paie RH, comptabilité et chiffre d'affaires, traces d'audit de sécurité SOC, anomalies de connexion, scorecards exécutifs. | `security_admin`, `ciso`, `dpo`, `executive`, `finance_controller`, `hr_director` |

### 2.2 Règle formelle d'habilitation
L'accès à un chunk documentaire est accordé si et seulement si :
$$\text{sensitivity\_level} \le \text{clearance\_level}$$

* **Utilisateur Niveau 1 (Public) :** Accès exclusif aux documents de niveau 1.
* **Utilisateur Niveau 2 (Interne) :** Accès aux documents de niveaux 1 et 2 (exclusion absolue du niveau 3).
* **Utilisateur Niveau 3 (Restreint / Confidentiel) :** Accès à l'ensemble des documents (niveaux 1, 2 et 3).

### 2.3 Dérivation d'accréditation côté serveur (Sécurité de l'Identité)
Le système n'accorde **aucune confiance à un niveau d'accréditation fourni directement par le client**. Le modèle `UserContext` est instancié côté backend à partir des rôles certifiés de la session :
```python
# Dérivation sécurisée côté backend
user_context = UserContext.from_roles(
    user_id="alice.dupont",
    roles=["employee", "data_engineer"]  # Déduit clearance_level = 2
)
```
Si aucun rôle n'est reconnu ou fourni, le système applique le principe du moindre privilège et rétrograde l'accréditation au Niveau 1 (Public).

---

## 3. Stratégie de Filtrage et Garanties de Sécurité (*Defense in Depth*)

La sécurité repose sur deux barrières concentriques indépendantes :

### 3.1 Barrière 1 : Préfiltrage natif dans le moteur ChromaDB
Le filtre de métadonnées est transmis directement à l'appel vectoriel :
```python
where = {"sensitivity_level": {"$lte": user_context.clearance_level}}
```
* **Rôle :** ChromaDB filtre l'ensemble des candidats selon les métadonnées indexées, garantissant que les résultats retournés satisfont la contrainte.
* **Prévention de l'éviction sémantique :** Contrairement à un post-filtrage naïf (qui extrairait les plus proches voisins globaux avant d'en masquer certains et risquerait de renvoyer une liste vide à un profil public), le préfiltrage garantit que les `top_k` documents renvoyés sont les plus pertinents *parmi ceux autorisés*.
* **Précision technique :** Les algorithmes de graphe approximatif (HNSW) peuvent naviguer à travers la structure d'indexation pour identifier les voisins, mais la couche de préfiltrage vectoriel assure que la liste finale extraite de l'index ne contient que les nœuds validant le filtre.

### 3.2 Barrière 2 : Validation défensive applicative (*Zero Trust / Default-Deny*)
Avant toute construction de l'objet résultat `RetrievalResult`, la méthode `RBACManager.is_access_granted` réexamine chaque chunk individuel extrait :
1. **Contrôle d'intégrité :** La présence du dictionnaire de métadonnées et de la clé `sensitivity_level` est exigée.
2. **Typage strict :** La valeur doit être un entier strict (`int`, exclusion formelle des types booléens et textuels).
3. **Plage de validité :** La valeur doit appartenir à $\{1, 2, 3\}$.
4. **Vérification de l'inégalité :** $\text{sensitivity\_level} \le \text{clearance\_level}$.
5. **Politique de Default-Deny :** Tout chunk présentant une anomalie (champ manquant, type inattendu, valeur corrompue) est **immédiatement rejeté et ignoré**.

---

## 4. Métrique Géométrique et Calcul du Score de Similarité

ChromaDB opère dans l'espace métrique cosinus configuré lors de la Phase 6 (`hnsw:space: "cosine"`).

### 4.1 Formule de calcul
Pour deux vecteurs L2-normalisés $u$ et $v$, ChromaDB calcule la distance cosinus brute :
$$D_{\text{cosine}}(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2} = 1 - \cos(\theta)$$

Le Retriever restitue le score de similarité cosinus exact :
$$\text{similarity\_score} = 1 - D_{\text{cosine}}(u, v) = \cos(\theta)$$

### 4.2 Domaines mathématiques théoriques
* **Distance cosinus :** $D \in [0.0, 2.0]$.
  - $D = 0.0$ : Vecteurs identiques (alignement parfait, $\theta = 0$).
  - $D = 1.0$ : Vecteurs orthogonaux ($\theta = 90^\circ$).
  - $D = 2.0$ : Vecteurs diamétralement opposés ($\theta = 180^\circ$).
* **Score de similarité cosinus :** $S \in [-1.0, 1.0]$.
  - $S = 1.0$ : Similarité maximale.
  - $S = 0.0$ : Indépendance sémantique (orthogonalité).
  - $S = -1.0$ : Opposition sémantique absolue.

> [!NOTE]
> Le score de similarité n'est pas borné artificiellement dans $[0, 1]$ afin de préserver l'exactitude mathématique de l'espace vectoriel cosinus. Sur des textes naturels encodés par `multilingual-e5-small`, les scores observés se situent généralement entre $0.40$ et $0.95$.

---

## 5. Spécification des Interfaces et Format des Résultats

### 5.1 Validation de `top_k`
Le paramètre `top_k` fait l'objet d'une validation stricte (`SemanticRetriever._validate_top_k`) :
- Doit être un entier strict (rejet des chaînes `"3"`, des flottants `3.0` et des booléens `True`).
- Doit être strictement supérieur à 0 ($k \ge 1$).
- Doit respecter la borne maximale de sécurité : $k \le 50$ (`MAX_TOP_K`).
- En cas de collection vide ou de base contenant moins de documents que demandé, le nombre de résultats est automatiquement ajusté (*clamping*) : $n = \min(k, \text{total\_documents})$.

### 5.2 Structure du résultat (`RetrievalResult`)
L'objet retourné est immuable et typé :
- `query` : Texte nettoyé de la requête utilisateur.
- `user_id` : Identifiant de l'appelant.
- `clearance_level` : Niveau d'accréditation vérifié (1, 2 ou 3).
- `chunks` : Liste ordonnée de `RetrievedChunk` autorisés.
- `total_retrieved` : Nombre de chunks retournés.
- `top_k_requested` : Valeur $k$ demandée.
- `execution_time_ms` : Durée totale de traitement (vectorisation + interrogation + validation) en millisecondes.
- `filter_applied` : Clause de filtrage ChromaDB auditée.

### 5.3 Méthode `to_context_str()` pour la Phase 8
La méthode `RetrievalResult.to_context_str()` formate directement le bloc textuel documentaire prêt à être inséré dans le prompt du LLM :
```text
[Document 1 | ID: DS-001_str_c0_19aa323fee27 | Ref: DOC-CAT-001 | Source: data/raw/csv/dataset_catalog.csv | Sensibilité: 1]
Dataset : iot_telemetry_gateway_raw (Identifiant : DS-001)
Description : Télémétrie brute des passerelles IoT industrielles connectées...
```

---

## 6. Journalisation Sécurisée (*Sanitized Audit Logging*)

Afin de respecter les normes de conformité et de protection des données :
- Les rejets d'accès (dépassement de sensibilité) et les anomalies de métadonnées sont journalisés au niveau `WARNING`.
- Les entrées de journal contiennent **uniquement** des métadonnées techniques : `chunk_id`, `document_id`, `doc_type`, le niveau requis et le niveau de l'utilisateur.
- **Aucun extrait textuel, contenu de document ni mot-clé confidentiel n'est écrit dans les journaux.**

---

## 7. Exemples d'Exécution et Commandes PowerShell

### 7.1 Lancement des tests automatisés (pytest)
```powershell
.\.venv\Scripts\pytest.exe tests/test_retriever_rbac.py -v
```
*(Exécution complète des 17 tests unitaires et de sécurité en moins de 2 secondes avec base ChromaDB isolée en mémoire temporaire).*

### 7.2 Démonstration CLI du Retriever sur le corpus réel (142 chunks)
```powershell
.\.venv\Scripts\python.exe -m src.retrieval.semantic_retriever
```
*Le script exécute successivement 3 requêtes types pour des profils Public, Interne et Restreint, démontrant le filtrage étanche sur les 142 chunks réels.*

---

## 8. Limites Connues et Perspectives pour la Phase 8

1. **Absence de re-ranking cross-encoder :** Le classement repose actuellement sur la similarité bi-encoder pure (`intfloat/multilingual-e5-small`). L'ajout d'un re-ranker pourra être étudié pour affiner la pertinence fine du top-3.
2. **Authentification simulée :** L'identité et les rôles sont simulés côté serveur via `UserContext.from_roles()`. Dans un déploiement d'entreprise complet, ce contexte sera alimenté par un token JWT validé par la passerelle API ou le SSO d'entreprise.
3. **Transition Phase 8 :** La Phase 8 connectera directement `RetrievalResult.to_context_str()` au prompt d'inférence d'Ollama (`llama3.2`), complétant ainsi la boucle RAG sécurisée de bout en bout.

