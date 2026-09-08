"""Benchmark reproductible de la récupération (retrieval) du RAG.

Le README documente une limite observée sur 2 questions ad hoc (recherche
sémantique par défaut de ChromaDB, orientée anglais, qui ne remonte pas
toujours l'article le plus pertinent sur une question posée en français).
Ce script transforme cette observation ponctuelle en mesure reproductible sur
un petit jeu de questions représentatives, chacune associée au(x) titre(s)
d'article jugé(s) pertinent(s) par relecture manuelle du corpus (pas une
vérité terrain officielle : un jugement de pertinence documenté, comme le
reste de la curation du corpus, voir curation_log.csv).

Métriques :
- rang du premier article attendu dans les résultats (1 = meilleur), None si
  absent du top K_MAX ;
- present_at_3 : l'article attendu est-il dans les 3 premiers résultats
  (n_results par défaut utilisé par le chatbot, voir chatbot.py) ?
- recall@3 (fraction de questions où c'est le cas) et MRR@10 (mean reciprocal
  rank, 0 si absent du top 10) sur l'ensemble du benchmark.

Usage:
    python -m chatbot.evaluate_retrieval
"""
import json
import sys
from pathlib import Path

# Certains titres du corpus contiennent des caractères unicode hors de la
# page de code par défaut d'une console Windows (cp1252) -- reconfigurer en
# UTF-8 évite un crash à l'affichage plutôt que de simplifier les titres.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
from index_articles import chercher

OUT_PATH = Path(__file__).parent / "retrieval_eval_results.json"
K_MAX = 10        # profondeur de recherche pour le calcul du rang / MRR
K_RECALL = 3      # profondeur utilisée par le chatbot en production (n_articles=3)

# Chaque entrée : une question représentative des usages réels de l'app, et le(s)
# titre(s) d'article jugé(s) pertinent(s) pour y répondre (au moins un doit être
# retrouvé). Les paires FR/EN sur la même question (#2/#3) documentent
# explicitement l'écart de récupération français/anglais déjà relevé dans le README.
BENCHMARK = [
    {
        "categorie": "deep_learning_segmentation",
        "question": "What is the U-Net architecture used for biomedical image segmentation?",
        "titres_attendus": ["U-Net: Convolutional Networks for Biomedical Image Segmentation"],
    },
    {
        "categorie": "comparaison_dice_unet (EN)",
        "question": "Is a Dice score of 0.88 low compared to other U-Net studies on the scratch assay?",
        "titres_attendus": [
            "An automated in vitro wound healing microscopy image analysis approach utilizing U-net-based deep learning methodology"
        ],
    },
    {
        "categorie": "comparaison_dice_unet (FR)",
        "question": "Mon Dice de 0,882 est-il faible comparé à d'autres études U-Net sur scratch assay ?",
        "titres_attendus": [
            "An automated in vitro wound healing microscopy image analysis approach utilizing U-net-based deep learning methodology"
        ],
    },
    {
        "categorie": "metrics_evaluation",
        "question": "What metrics should I use to evaluate medical image segmentation quality?",
        "titres_attendus": [
            "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool",
            "Metrics reloaded: recommendations for image analysis validation",
        ],
    },
    {
        "categorie": "metrics_evaluation",
        "question": "Why can the Hausdorff distance be unstable across segmentation runs?",
        "titres_attendus": [
            "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool",
            "Metrics reloaded: recommendations for image analysis validation",
        ],
    },
    {
        "categorie": "scratch_assay_principles",
        "question": "What are the limitations of the scratch assay for studying wound healing?",
        "titres_attendus": [
            "The In Vitro Wound‐Scratch Assay: Applications, Technical Advances, and Limitations in Wound Healing Research"
        ],
    },
    {
        "categorie": "deep_learning_segmentation",
        "question": "Does pretraining the encoder on ImageNet improve U-Net segmentation accuracy?",
        "titres_attendus": ["TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation"],
    },
    {
        "categorie": "automated_analysis",
        "question": "What software tools exist for automated scratch assay quantification, like ImageJ plugins?",
        "titres_attendus": [
            "An ImageJ plugin for the high throughput image analysis of in vitro scratch wound healing assays",
            "TScratch: a novel and simple software tool for automated analysis of monolayer wound healing assays",
        ],
    },
    {
        "categorie": "deep_learning_segmentation",
        "question": "How to train a deep learning segmentation model with only a few annotated images?",
        "titres_attendus": ["Embracing imperfect datasets: A review of deep learning solutions for medical image segmentation"],
    },
    {
        "categorie": "analyse_temporelle",
        "question": "Combien de points de mesure sont nécessaires pour caractériser la cinétique de fermeture d'une plaie ?",
        "titres_attendus": [
            "Study of Wound Healing Dynamics by Single Pseudo-Particle Tracking in Phase Contrast Images Acquired in Time-Lapse",
            "A novel magnet-based scratch method for standardisation of wound-healing assays",
        ],
    },
]


# Reformulations testées pour les questions qui échouent dans le benchmark ci-dessus
# (present_at_3 = False) : même intention de question, vocabulaire plus technique/
# biomédical repris directement des résumés d'articles ciblés (ex: "Dice similarity
# coefficient", "wound-scratch assay"), au lieu d'une formulation familière. Compare
# Recall@3/MRR avant/après pour vérifier si le levier est bien la formulation, pas
# la langue (voir README, section Base documentaire du RAG).
REFORMULATIONS = [
    {
        "categorie": "comparaison_dice_unet (EN)",
        "question_reformulee": (
            "Dice similarity coefficient comparison of U-Net deep learning models for "
            "in vitro wound healing scratch assay segmentation versus ImageJ and TScratch"
        ),
    },
    {
        "categorie": "comparaison_dice_unet (FR)",
        "question_reformulee": (
            "Coefficient de similarité de Dice (DSC) des modèles U-Net pour la "
            "segmentation d'images de scratch assay de cicatrisation, comparé aux "
            "outils ImageJ et TScratch"
        ),
    },
    {
        "categorie": "scratch_assay_principles",
        "question_reformulee": (
            "In vitro wound-scratch assay: technical advances and limitations in "
            "wound healing research"
        ),
    },
    {
        "categorie": "analyse_temporelle",
        "question_reformulee": (
            "Suivi en time-lapse de la dynamique de fermeture de plaie par "
            "pseudo-particle tracking en scratch assay"
        ),
    },
]


def rank_of_expected(returned_titles: list[str], titres_attendus: list[str]) -> int | None:
    """Rang (1-indexé) du premier titre attendu trouvé dans les résultats retournés,
    dans l'ordre où ChromaDB les classe (donc déjà trié par pertinence décroissante).
    None si aucun des titres attendus n'apparaît."""
    attendus_norm = {t.strip().lower() for t in titres_attendus}
    for rank, titre in enumerate(returned_titles, start=1):
        if titre.strip().lower() in attendus_norm:
            return rank
    return None


def evaluate_question(entry: dict, hybride: bool = False) -> dict:
    articles = chercher(entry["question"], n=K_MAX, hybride=hybride)
    returned_titles = [a["titre"] for a in articles]
    rank = rank_of_expected(returned_titles, entry["titres_attendus"])
    return {
        "categorie": entry["categorie"],
        "question": entry["question"],
        "titres_attendus": entry["titres_attendus"],
        "rang_trouve": rank,
        "present_at_3": rank is not None and rank <= K_RECALL,
        "reciprocal_rank": (1.0 / rank) if rank is not None else 0.0,
        "top_3_retournes": returned_titles[:3],
    }


def evaluate_reformulations(benchmark_resultats: list[dict]) -> list[dict]:
    """Ré-interroge chaque question reformulée (REFORMULATIONS) et compare son rang/
    présence au top 3 à ceux de la question originale correspondante, pour isoler
    l'effet de la formulation à intention de question identique."""
    original_par_categorie = {r["categorie"]: r for r in benchmark_resultats}
    entry_par_categorie = {e["categorie"]: e for e in BENCHMARK}

    comparaisons = []
    for reform in REFORMULATIONS:
        original = original_par_categorie[reform["categorie"]]
        entry = entry_par_categorie[reform["categorie"]]

        articles = chercher(reform["question_reformulee"], n=K_MAX)
        returned_titles = [a["titre"] for a in articles]
        rank = rank_of_expected(returned_titles, entry["titres_attendus"])

        comparaisons.append({
            "categorie": reform["categorie"],
            "question_originale": entry["question"],
            "question_reformulee": reform["question_reformulee"],
            "rang_original": original["rang_trouve"],
            "rang_reformule": rank,
            "present_at_3_original": original["present_at_3"],
            "present_at_3_reformule": rank is not None and rank <= K_RECALL,
            "top_3_retournes_reformule": returned_titles[:3],
        })
    return comparaisons


def main():
    resultats = [evaluate_question(entry) for entry in BENCHMARK]

    n = len(resultats)
    recall_at_3 = sum(r["present_at_3"] for r in resultats) / n
    mrr_at_10 = sum(r["reciprocal_rank"] for r in resultats) / n

    print(f"Benchmark de retrieval RAG : {n} questions\n")
    for r in resultats:
        statut = f"rang {r['rang_trouve']}" if r["rang_trouve"] is not None else f"absent du top {K_MAX}"
        marque = "OK " if r["present_at_3"] else "X  "
        print(f"[{marque}] ({r['categorie']}) {statut} -- {r['question']}")
        if not r["present_at_3"]:
            print(f"       attendu: {r['titres_attendus']}")
            print(f"       obtenu (top 3): {r['top_3_retournes']}")

    print(f"\nRecall@{K_RECALL} : {recall_at_3:.2%} ({sum(r['present_at_3'] for r in resultats)}/{n})")
    print(f"MRR@{K_MAX}      : {mrr_at_10:.3f}")

    comparaisons = evaluate_reformulations(resultats)
    n_reform = len(comparaisons)
    recall_avant = sum(c["present_at_3_original"] for c in comparaisons) / n_reform
    recall_apres = sum(c["present_at_3_reformule"] for c in comparaisons) / n_reform

    print(f"\n--- Expérience de reformulation ({n_reform} questions qui échouaient) ---")
    for c in comparaisons:
        rang_avant = c["rang_original"] if c["rang_original"] is not None else f">{K_MAX}"
        rang_apres = c["rang_reformule"] if c["rang_reformule"] is not None else f">{K_MAX}"
        marque = "OK " if c["present_at_3_reformule"] else "X  "
        print(f"[{marque}] ({c['categorie']}) rang {rang_avant} -> {rang_apres}")
        print(f"       reformulée : {c['question_reformulee']}")
        if not c["present_at_3_reformule"]:
            print(f"       obtenu (top 3): {c['top_3_retournes_reformule']}")

    print(f"\nRecall@{K_RECALL} sur ces {n_reform} questions : {recall_avant:.0%} (original) "
          f"-> {recall_apres:.0%} (reformulé)")

    # Recherche hybride (embeddings + BM25 mots-clés, voir index_articles.py) : même
    # benchmark, réinterrogé avec hybride=True, pour valider avant/après si cette
    # piste (documentée en Perspectives du README) mérite d'être adoptée par défaut
    # dans chatbot.py -- pas supposée bénéfique sur simple lecture du code.
    resultats_hybrides = [evaluate_question(entry, hybride=True) for entry in BENCHMARK]
    recall_at_3_hybride = sum(r["present_at_3"] for r in resultats_hybrides) / n
    mrr_at_10_hybride = sum(r["reciprocal_rank"] for r in resultats_hybrides) / n

    print(f"\n--- Recherche hybride (embeddings + BM25) ---")
    for original, hybride in zip(resultats, resultats_hybrides):
        rang_avant = original["rang_trouve"] if original["rang_trouve"] is not None else f">{K_MAX}"
        rang_apres = hybride["rang_trouve"] if hybride["rang_trouve"] is not None else f">{K_MAX}"
        marque = "OK " if hybride["present_at_3"] else "X  "
        if original["rang_trouve"] != hybride["rang_trouve"]:
            print(f"[{marque}] ({hybride['categorie']}) rang {rang_avant} -> {rang_apres} -- {hybride['question']}")

    print(f"\nRecall@{K_RECALL} : {recall_at_3:.2%} (embeddings seuls) -> {recall_at_3_hybride:.2%} (hybride)")
    print(f"MRR@{K_MAX}      : {mrr_at_10:.3f} (embeddings seuls) -> {mrr_at_10_hybride:.3f} (hybride)")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "k_recall": K_RECALL,
            "k_max": K_MAX,
            "n_questions": n,
            "recall_at_k": recall_at_3,
            "mrr_at_k_max": mrr_at_10,
            "per_question": resultats,
            "reformulation_experiment": {
                "n_questions": n_reform,
                "recall_at_k_original": recall_avant,
                "recall_at_k_reformule": recall_apres,
                "comparaisons": comparaisons,
            },
            "hybrid_experiment": {
                "recall_at_k_embeddings": recall_at_3,
                "recall_at_k_hybride": recall_at_3_hybride,
                "mrr_at_k_max_embeddings": mrr_at_10,
                "mrr_at_k_max_hybride": mrr_at_10_hybride,
                "per_question_hybride": resultats_hybrides,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nRésultats -> {OUT_PATH}")


if __name__ == "__main__":
    main()
