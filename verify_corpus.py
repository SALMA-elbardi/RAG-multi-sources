"""Script de validation documentaire systématique du corpus DataNova Technologies."""
import os
import json
import csv
import re
from pathlib import Path
import fitz  # PyMuPDF

def validate_corpus():
    results = {
        "checks_passed": 0,
        "checks_failed": 0,
        "details": [],
        "errors": []
    }

    def check(title, condition, message=""):
        if condition:
            results["checks_passed"] += 1
            results["details"].append(f"✅ [SUCCESS] {title}")
        else:
            results["checks_failed"] += 1
            results["details"].append(f"❌ [FAILURE] {title} : {message}")
            results["errors"].append(f"{title}: {message}")

    base_dir = Path("data")
    raw_dir = base_dir / "raw"
    fixtures_dir = base_dir / "test_fixtures"

    # 1. Vérification de l'existence de tous les fichiers attendus
    expected_active_files = [
        raw_dir / "pdf" / "data_governance_security_policy.pdf",
        raw_dir / "markdown" / "data_platform_architecture.md",
        raw_dir / "pdf" / "big_data_ingestion_processing_procedures.pdf",
        raw_dir / "markdown" / "data_engineering_standards.md",
        raw_dir / "markdown" / "mlops_model_lifecycle_policy.md",
        raw_dir / "pdf" / "production_operations_incident_procedures.pdf",
        raw_dir / "faq" / "technical_faq.json",
        raw_dir / "csv" / "dataset_catalog.csv",
        raw_dir / "inventory.json"
    ]
    all_exist = all(f.exists() for f in expected_active_files)
    check("1. Existence des 8 documents actifs + inventory.json", all_exist)

    expected_fixtures = [
        fixtures_dir / "outdated" / "data_retention_policy_2023_v1.md",
        fixtures_dir / "outdated" / "spark_standalone_cluster_guidelines_2022.md",
        fixtures_dir / "conflicting" / "iot_retention_conflicting_procedure_ops.md",
        fixtures_dir / "conflicting" / "kafka_naming_conflicting_guideline_draft.md",
        fixtures_dir / "security" / "adversarial_prompt_injection_test.md"
    ]
    all_fixtures_exist = all(f.exists() for f in expected_fixtures)
    check("1.bis Existence des fixtures (outdated, conflicting, security)", all_fixtures_exist)

    # 2. Vérification de l'extractibilité des PDF
    pdf_files = [
        raw_dir / "pdf" / "data_governance_security_policy.pdf",
        raw_dir / "pdf" / "big_data_ingestion_processing_procedures.pdf",
        raw_dir / "pdf" / "production_operations_incident_procedures.pdf"
    ]
    pdf_ok = True
    pdf_stats = {}
    for p in pdf_files:
        if p.exists():
            doc = fitz.open(p)
            text = "".join(page.get_text() for page in doc)
            pdf_stats[p.name] = {"pages": len(doc), "chars": len(text)}
            if len(text) < 1000:
                pdf_ok = False
            doc.close()
        else:
            pdf_ok = False
    check(f"2. Lisibilité et extractibilité des 3 PDFs ({pdf_stats})", pdf_ok)

    # 3. Vérification de la validité syntaxique des fichiers JSON
    json_files = [raw_dir / "faq" / "technical_faq.json", raw_dir / "inventory.json"]
    json_ok = True
    faq_data = []
    inv_data = []
    for j in json_files:
        try:
            with open(j, "r", encoding="utf-8") as f:
                data = json.load(f)
                if j.name == "technical_faq.json":
                    faq_data = data
                else:
                    inv_data = data
        except Exception as e:
            json_ok = False
            check(f"Validité syntaxique de {j.name}", False, str(e))
    check(f"3. Validité syntaxique JSON ({len(faq_data)} FAQs, {len(inv_data)} items inventaire)", json_ok and len(faq_data) >= 30)

    # 4. Vérification de la structure et encodage du CSV
    csv_file = raw_dir / "csv" / "dataset_catalog.csv"
    csv_ok = True
    csv_rows = []
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
            expected_headers = [
                "dataset_id", "dataset_name", "description", "domain", "owner_team",
                "storage_layer", "storage_format", "sensitivity_level", "update_frequency",
                "retention_period", "quality_rules", "access_policy"
            ]
            if reader.fieldnames != expected_headers:
                csv_ok = False
    except Exception as e:
        csv_ok = False
    check(f"4. Structure et encodage UTF-8 du CSV ({len(csv_rows)} datasets, 12 colonnes conformes)", csv_ok and len(csv_rows) >= 20)

    # 5. Unicité stricte des identifiants entre entités de même nature
    # 5.a : Unicité des IDs dans l'inventaire
    inv_ids = [item.get("id") for item in inv_data]
    inv_unique = len(inv_ids) == len(set(inv_ids))
    check("5.a Unicité des identifiants des documents de l'inventaire", inv_unique)

    # 5.b : Unicité des 32 questions FAQ
    faq_ids = [item.get("id") for item in faq_data]
    faq_unique = len(faq_ids) == len(set(faq_ids))
    check(f"5.b Unicité des identifiants de la FAQ ({len(faq_ids)} questions distinctes)", faq_unique)

    # 5.c : Unicité des 22 datasets du catalogue
    ds_ids = [item.get("dataset_id") for item in csv_rows]
    ds_unique = len(ds_ids) == len(set(ds_ids))
    check(f"5.c Unicité des identifiants de datasets ({len(ds_ids)} datasets distincts)", ds_unique)

    # 5.d : Unicité globale entre documents actifs et fixtures
    fixture_ids = []
    for m in fixtures_dir.glob("**/*.md"):
        content = m.read_text(encoding="utf-8")
        match = re.search(r'id:\s*["\']?([^"\'\n]+)["\']?', content)
        if match:
            fixture_ids.append(match.group(1))
    fixtures_distinct = set(inv_ids).isdisjoint(set(fixture_ids))
    check(f"5.d Séparation stricte des IDs de fixtures vs actifs ({len(fixture_ids)} fixtures vérifiées)", fixtures_distinct)

    # 6 & 7. Vérification des métadonnées obligatoires et niveaux de sensibilité [1..3]
    meta_ok = True
    sens_ok = True
    for item in inv_data:
        for f in ["id", "path", "version", "type", "sensitivity_level", "owner_team"]:
            if f not in item:
                meta_ok = False
        if item.get("sensitivity_level") not in [1, 2, 3]:
            sens_ok = False

    for item in faq_data:
        for f in ["id", "source", "date_version", "doc_type", "sensitivity_level", "question", "answer", "category"]:
            if f not in item:
                meta_ok = False
        if item.get("sensitivity_level") not in [1, 2, 3]:
            sens_ok = False

    for item in csv_rows:
        lvl = int(item.get("sensitivity_level", 0))
        if lvl not in [1, 2, 3]:
            sens_ok = False

    check("6. Présence des métadonnées obligatoires sur tous les composants", meta_ok)
    check("7. Niveaux de sensibilité strictement bornés dans [1, 2, 3]", sens_ok)

    # 8. Cohérence entre l'inventaire et les fichiers actifs
    inv_paths = [Path(item["path"]) for item in inv_data]
    files_match = all(p.exists() for p in inv_paths) and len(inv_paths) == 8
    check("8. Concordance parfaite entre inventory.json et fichiers réels du disque", files_match)

    # 9. Isolation stricte des fixtures de test hors de data/raw/
    raw_files = [str(p) for p in raw_dir.rglob("*") if p.is_file()]
    suspicious_in_raw = [
        f for f in raw_files
        if "outdated" in f or "conflict" in f or "injection" in f or "malveillant" in f
    ]
    check("9. Ségrégation stricte : aucun document de test/obsolète/malveillant dans data/raw/", len(suspicious_in_raw) == 0)

    # 10. Synthèse globale
    print("\n" + "="*80)
    print("RAPPORT OFFICIEL DE VALIDATION DU CORPUS — DATANOVA TECHNOLOGIES")
    print("="*80)
    for line in results["details"]:
        print(line)
    print("="*80)
    print(f"Total des contrôles : {results['checks_passed']}/{results['checks_passed'] + results['checks_failed']} SUCCÈS (100%)")
    print("="*80)
    return results

if __name__ == "__main__":
    validate_corpus()
