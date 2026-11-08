# Architecture & Documentation du Pipeline d'Ingestion Documentaire Multi-Sources

**Projet :** Système RAG Multi-Sources & Sécurité Documentaire  
**Entreprise Fictive :** DataNova Technologies  
**Phase :** Phase 4 — Ingestion Multi-Sources, Normalisation & Validation  
**Statut :** Opérationnel et Testé  

---

## 1. Vision et Objectifs de l'Ingestion

Le pipeline d'ingestion documentaire constitue la porte d'entrée unique de l'ensemble des connaissances techniques et organisationnelles de **DataNova Technologies**.

Ses objectifs clés sont :
1. **Multi-sources unifié :** Charger de façon homogène les documents au format PDF, Markdown, JSON (FAQ) et CSV (catalogue de données).
2. **Normalisation standardisée :** Transformer chaque flux en une collection en mémoire d'objets `NormalizedDocument` compatibles avec l'écosystème LangChain (`page_content` et `metadata`).
3. **Gouvernance et traçabilité :** Préserver la provenance exacte de chaque élément à travers un jeu de 5 métadonnées universelles obligatoires issues d'[inventory.json](file:///c:/Users/Salma/Desktop/RAG-multi-sources/data/raw/inventory.json).
4. **Sécurité et compatibilité vectorielle (ChromaDB) :** Garantir que les métadonnées sont composées exclusivement de types primitifs (`str`, `int`, `float`, `bool`) et que les niveaux d'accréditation RBAC sont strictement validés ($\in \{1, 2, 3\}$).
5. **Ségrégation stricte des jeux de test :** Assurer qu'aucun document issu de `data/test_fixtures/` (obsolète, contradictoire ou malveillant) ne puisse pénétrer dans le corpus actif.

---

## 2. Architecture Modulaire du Code (`src/ingestion/`)

```text
src/
└── ingestion/
    ├── __init__.py              # Exposition publique et rétrocompatibilité
    ├── models.py                # Modèles NormalizedDocument et IngestionReport
    ├── metadata_manager.py      # Chargement et réconciliation d'inventory.json
    ├── pdf_loader.py            # Extraction des PDF page par page avec page_number
    ├── markdown_loader.py       # Extraction Markdown structuré et parsing frontmatter
    ├── json_loader.py           # Découpage unitaire des Q/R de la FAQ technique
    ├── csv_loader.py            # Modélisation sémantique des 22 datasets sur 12 colonnes
    ├── document_validator.py    # Contrôle qualité (bloquants, avertissements, ChromaDB)
    ├── loader_factory.py        # Dispatcher avec cache d'instances de loaders
    └── ingestion_pipeline.py    # Orchestrateur central et reporting dynamique
```

---

## 3. Schéma de Données Unifié (`NormalizedDocument`)

Chaque document normalisé produit par les loaders et validé par le pipeline respecte la structure suivante :

```python
class NormalizedDocument(BaseModel):
    id: str                          # Identifiant unique et stable de l'élément (ex: DOC-SEC-001#p1, FAQ-DE-001, DS-001)
    page_content: str                # Texte extrait, nettoyé et signifiant
    metadata: Dict[str, Union[str, int, float, bool]] # Métadonnées plates (ChromaDB compliant)
```

### 3.1. Métadonnées Universelles Obligatoires
Présentes sur **100 % des documents validés** :
* `document_id` (`str`) : Identifiant du document source parent dans l'inventaire officiel (ex: `DOC-SEC-001`, `DOC-ARCH-001`, `DOC-FAQ-001`, `DOC-CAT-001`).
* `source` (`str`) : Chemin relatif normalisé (ex: `data/raw/pdf/data_governance_security_policy.pdf`).
* `date_version` (`str`) : Numéro de version ou date de publication certifiée.
* `doc_type` (`str`) : `pdf`, `markdown`, `faq` ou `csv`.
* `sensitivity_level` (`int`) : Niveau de sensibilité RBAC ($1$: Public, $2$: Interne restreint, $3$: Confidentiel).

### 3.2. Métadonnées Spécifiques par Format
* **PDF :** `page_number` (`int`), `total_pages` (`int`), `title` (`str`), `owner_team` (`str`).
* **FAQ :** `faq_id` (`str`), `category` (`str`), `title` (`str`).
* **CSV :** `dataset_id` (`str`), `domain` (`str`), `owner_team` (`str`), `storage_layer` (`str`), `storage_format` (`str`), `update_frequency` (`str`), `retention_period` (`str`), `access_policy` (`str`).
* **Markdown :** `title` (`str`), `owner_team` (`str`).

---

## 4. Fonctionnement Détaillé des Loaders

### 4.1. PDFLoader (`pdf_loader.py`)
* S'appuie sur la bibliothèque standard `pypdf`.
* Parcourt chaque page du document PDF et génère un document normalisé distinct avec identifiant stable `<document_id>#p<page_number>`.
* **Zéro perte silencieuse :** Si une page contient moins de 10 caractères utiles (page blanche ou non extractible), elle n'est pas simplement écartée : elle est consignée avec le numéro de page, le nom du fichier et la raison dans `IngestionReport.ignored_pages_details`.

### 4.2. MarkdownLoader (`markdown_loader.py`)
* Extrait le frontmatter YAML d'en-tête (délimité par `---`).
* Invoque `MetadataManager.detect_discrepancies()` pour vérifier la concordance avec `inventory.json`. Toute divergence (ex : de version ou de niveau de sensibilité) est tracée sans écrasement masqué.
* Préserve l'arborescence des sections, titres (`#`, `##`), tableaux et blocs de code (ex: diagrammes Mermaid).

### 4.3. JSONLoader (`json_loader.py`)
* Découpe le fichier `technical_faq.json` en autant de documents indépendants qu'il y a de questions-réponses (32 documents).
* Formate le contenu textuel sous la forme :  
  `Question : <question>\nRéponse : <answer>`.
* Conserve la catégorie technique et l'identifiant individuel (`FAQ-xxx`).

### 4.4. CSVLoader (`csv_loader.py`)
* Parse le catalogue `dataset_catalog.csv` en UTF-8.
* Chaque ligne devient un document autonome modélisé en langage naturel pour la recherche sémantique :
  ```text
  Dataset : <dataset_name> (Identifiant : <dataset_id>)
  Description : <description>
  Domaine d'activité : <domain> | Équipe responsable : <owner_team>
  Stockage Lakehouse : Couche <storage_layer> (Format de stockage : <storage_format>)
  Fréquence de mise à jour : <update_frequency> | Rétention des données : <retention_period>
  Règles et contrôles qualité : <quality_rules>
  Politique et habilitations d'accès : <access_policy> (Niveau de sensibilité : <sensitivity_level>)
  ```
* Conserve **l'intégralité des 12 colonnes** dans les métadonnées pour permettre un filtrage analytique précis.

---

## 5. Règles de Validation & Sécurité (`DocumentValidator`)

Le validateur applique les contrôles suivants :
1. **Intégrité textuelle :** Rejette tout document dont le `page_content` est vide ou composé uniquement d'espaces blancs.
2. **Complétude des métadonnées :** Rejette tout document dépourvu d'au moins l'une des 5 métadonnées obligatoires universelles.
3. **Contrôles spécifiques par format :** Rejette un PDF sans `page_number`, une FAQ sans `faq_id` ou `category`, un dataset sans `dataset_id` ou `domain`.
4. **Validité RBAC :** Rejette tout document dont le `sensitivity_level` n'est pas un entier $\in \{1, 2, 3\}$.
5. **Garde-fou ChromaDB :** Rejette toute métadonnée contenant des types non primitifs (listes, dictionnaires).
6. **Non-destructivité :** Les métadonnées originales ne sont jamais modifiées ; aucune valeur manquante n'est inventée pour forcer artificiellement la validation.

---

## 6. Ségrégation Stricte et Résilience de l'Orchestrateur

* **Exclusion des fixtures :** L'orchestrateur refuse formellement de charger tout fichier contenant dans son chemin les segments `test_fixtures`, `outdated`, `conflicting` ou `security`.
* **Exclusion d'inventory.json :** Le fichier d'inventaire est utilisé comme source de vérité des métadonnées et n'est jamais indexé comme document textuel.
* **Résilience d'exécution :** Si un fichier du corpus actif est corrompu ou illisible, l'erreur est capturée et enregistrée dans `IngestionReport.errors`, et le pipeline poursuit le chargement des autres fichiers sains.

---

## 7. Commandes d'Exécution et Validation Manuelle

Toutes les opérations peuvent être lancées et vérifiées manuellement via PowerShell :

```powershell
# 1. Exécution du pipeline d'ingestion et affichage du rapport statistique
.\.venv\Scripts\python.exe -m src.ingestion.ingestion_pipeline

# 2. Exécution de la suite complète des tests unitaires d'ingestion
.\.venv\Scripts\pytest.exe tests/test_ingestion.py -v
```

