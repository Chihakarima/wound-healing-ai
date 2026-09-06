"""Reconstruit chatbot/articles_cicatrisation.json en rééquilibrant la base RAG :
- garde un petit noyau d'études biologiques (traitements/molécules) déjà présentes,
- garde/complète les articles sur les principes et limites du scratch assay,
- garde/complète les articles sur l'automatisation et la quantification d'image,
- ajoute des articles sur la segmentation par deep learning (absents jusqu'ici),
- ajoute des articles sur les métriques d'évaluation (Dice/IoU/Hausdorff),
en attachant à chaque article une "category" et des "topics" pour documenter le
rôle de chaque source dans le RAG.

Produit aussi chatbot/curation_log.csv (Titre | Statut | Raison) pour tracer
explicitement ce qui a été gardé/rejeté et pourquoi, plutôt que de faire une
sélection silencieuse.

Usage:
    python chatbot/curate_and_merge.py
"""
import csv
import json
from pathlib import Path

ANCIEN_PATH = Path(__file__).parent / "articles_cicatrisation.json"
NOUVEAUX_PATH = Path(__file__).parent / "articles_nouveaux_categories.json"
OUT_PATH = Path(__file__).parent / "articles_cicatrisation.json"
LOG_PATH = Path(__file__).parent / "curation_log.csv"

TOPICS_PAR_CATEGORIE = {
    "scratch_assay_principles": ["scratch_assay", "principes", "limites"],
    "automated_analysis": ["scratch_assay", "automatisation", "analyse_image", "reproductibilite"],
    "deep_learning_segmentation": ["deep_learning", "segmentation", "microscopie"],
    "metrics_evaluation": ["metriques", "evaluation", "quantification"],
    "biological_application": ["scratch_assay", "application_biologique", "traitement"],
}

# Articles à conserver depuis l'ancienne base, avec leur catégorie.
TITRES_A_GARDER = {
    "The In Vitro Wound‐Scratch Assay: Applications, Technical Advances, and Limitations in Wound Healing Research":
        "scratch_assay_principles",
    "Therapeutic candidates for keloid scars identified by qualitative review of scratch assay research for wound healing":
        "scratch_assay_principles",
    "Automated In Vitro Wound Healing Assay": "automated_analysis",
    "Optimized Scratch Assay for In Vitro Testing of Cell Migration with an Automated Optical Camera":
        "automated_analysis",
    "An on-chip wound healing assay fabricated by xurography for evaluation of dermal fibroblast cell migration and wound closure":
        "automated_analysis",
    "Migratory Metrics of Wound Healing: A Quantification Approach for in vitro Scratch Assays":
        "metrics_evaluation",
    "Trehalose promotes wound healing in vitro by enhancing the migration of human keratinocytes via the VEGF/JNK/PI3K pathway":
        "biological_application",
    "PDGF-AA loaded photo-crosslinked chitosan-based hydrogel for promoting wound healing.":
        "biological_application",
    "Polyethylene Glycol Loxenatide Accelerates Diabetic Wound Healing by Downregulating Systemic Inflammation and Improving Endothelial Progenitor Cell Functions":
        "biological_application",
    "Exosomes from human induced pluripotent stem cells-derived keratinocytes accelerate burn wound healing through miR-762 mediated promotion of keratinocytes and endothelial cells migration":
        "biological_application",
}

# Titres récupérés par fetch_articles_by_category.py jugés hors sujet après relecture
# manuelle du résumé (pas seulement un mot-clé qui matche) : {titre: raison}.
EXCLUSIONS_MANUELLES: dict[str, str] = {
    "Improvement in Transient Agarose Spot (TAS) Cell Migration Assay: Microplate-Based Detection and Evaluation":
        "Porte sur un autre type d'essai (Transient Agarose Spot), pas un scratch assay : "
        "moins directement transférable au projet malgré le mot-clé 'migration assay'.",
    "Trehalose promotes wound healing in vitro by enhancing the migration of human keratinocytes via the VEGF/JNK/PI3K pathway":
        "Doublon exact (même titre) avec l'article déjà retenu en catégorie biological_application ; "
        "resurgit ici car il mentionne aussi le scratch assay dans son résumé.",
    "Photobiomodulation and Wound Healing: Low-Level Laser Therapy at 661 nm in a Scratch Assay Keratinocyte Model":
        "Étude de traitement (photobiomodulation), pas une méthode d'automatisation/analyse d'image : "
        "récupérée par la requête à cause du mot-clé 'scratch assay', hors sujet pour cette catégorie.",
}

# Cibles de taille par catégorie (nombre max à conserver après fusion avec les nouveaux).
# deep_learning_segmentation et metrics_evaluation restent à 0 : les requêtes correspondantes
# ont épuisé leurs 6 tentatives sur l'API Semantic Scholar (429 persistant) sans résultat.
# Pas de contenu inventé pour combler ce manque : voir curation_log.csv pour le détail.
CIBLES = {
    "scratch_assay_principles": 3,
    "automated_analysis": 10,
    "deep_learning_segmentation": 0,
    "metrics_evaluation": 1,
    "biological_application": 4,
}


def _tag(article: dict, categorie: str) -> dict:
    article = dict(article)
    article["category"] = categorie
    article["topics"] = TOPICS_PAR_CATEGORIE[categorie]
    return article


def main():
    anciens = json.loads(ANCIEN_PATH.read_text(encoding="utf-8"))
    nouveaux_par_categorie = json.loads(NOUVEAUX_PATH.read_text(encoding="utf-8"))

    par_categorie: dict[str, list[dict]] = {c: [] for c in CIBLES}
    log_rows: list[tuple[str, str, str]] = []  # (titre, statut, raison)

    for a in anciens:
        cat = TITRES_A_GARDER.get(a["title"])
        if cat:
            par_categorie[cat].append(_tag(a, cat))
            log_rows.append((a["title"], "Gardé", f"Ancien article, catégorie {cat}"))
        else:
            log_rows.append((a["title"], "Rejeté",
                              "Étude de traitement/molécule hors du cœur du projet "
                              "(retirée lors du rééquilibrage vers segmentation/évaluation)"))

    for cat, articles in nouveaux_par_categorie.items():
        if cat not in par_categorie:
            continue
        titres_deja_presents = {a["title"].strip().lower() for a in par_categorie[cat]}
        for a in articles:
            titre = a["title"] or ""
            titre_norm = titre.strip().lower()
            if not titre_norm:
                continue
            raison_exclusion = EXCLUSIONS_MANUELLES.get(titre)
            if raison_exclusion:
                log_rows.append((titre, "Rejeté", raison_exclusion))
            elif titre_norm in titres_deja_presents:
                log_rows.append((titre, "Rejeté", "Doublon avec un article déjà retenu"))
            elif len(par_categorie[cat]) >= CIBLES[cat]:
                log_rows.append((titre, "Rejeté",
                                  f"Quota atteint pour la catégorie {cat} ({CIBLES[cat]} articles)"))
            else:
                par_categorie[cat].append(_tag(a, cat))
                titres_deja_presents.add(titre_norm)
                log_rows.append((titre, "Gardé", f"Nouvel article, catégorie {cat}"))

    resultat = [a for cat in CIBLES for a in par_categorie[cat]]

    OUT_PATH.write_text(json.dumps(resultat, ensure_ascii=False, indent=2), encoding="utf-8")

    with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Titre", "Statut", "Raison"])
        writer.writerows(log_rows)

    print(f"Total retenu: {len(resultat)} articles")
    for cat in CIBLES:
        print(f"  - {cat}: {len(par_categorie[cat])}/{CIBLES[cat]}")
    print(f"Journal de curation: {LOG_PATH} ({len(log_rows)} lignes)")


if __name__ == "__main__":
    main()
