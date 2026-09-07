"""Analyse d'erreur quantitative par catégorie, sur les 14 images de test.

Croise les métriques déjà calculées (`test_metrics.json` pour le U-Net,
`test_baseline_metrics.json` pour la baseline) avec deux critères objectifs et
reproductibles, plutôt qu'un classement visuel manuel :
- surface relative de la plaie, seuillée à 3 % (cf. section Limites connues du
  README : "performances dégradées sur les plaies quasi refermées (< ~3 %)") ;
- luminosité du champ (proxy des conditions d'éclairage/contraste, cf. image 76
  dans l'analyse d'erreur du README), comparée par z-score à la distribution du
  dataset complet (97 images), pas seulement aux 14 images de test.

Avec seulement 14 images de test, chaque catégorie ne contient que 2 à 8
images : ce tableau sert à repérer des cas particuliers et vérifier une
tendance déjà observée qualitativement, pas à établir un résultat statistique.
Le nombre d'images par catégorie (n) est toujours affiché à côté de chaque
moyenne pour ne pas laisser croire à une robustesse qu'un si petit échantillon
ne permet pas d'affirmer.

Usage:
    python -m src.error_analysis_categories
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from baseline import FOV_ERODE_KSIZE, field_of_view_mask
from dataset import find_image_path, load_ids

SPLITS_DIR = "data/splits"
PRED_DIR = "outputs/predictions"
QC_AREA_FRACTIONS = "outputs/qc_masks/_area_fractions.csv"
OUT_PER_IMAGE = "outputs/predictions/test_error_by_category.csv"
OUT_SUMMARY = "outputs/predictions/test_error_category_summary.csv"

AREA_THRESHOLD_PCT = 3.0     # seuil "quasi refermée", cf. Limites connues du README
BRIGHTNESS_Z_THRESHOLD = 1.5  # |z-score| au-delà duquel l'éclairage est jugé atypique


def load_area_fractions():
    area_frac = {}
    with open(QC_AREA_FRACTIONS) as f:
        for row in csv.reader(f):
            if row[0] == "id":
                continue
            area_frac[row[0]] = float(row[1])
    return area_frac


def field_brightness(stem):
    """Luminosité moyenne du champ du microscope (fond noir et marge de
    vignettage exclus), utilisée comme proxy objectif des conditions
    d'éclairage/contraste d'une image."""
    gray = cv2.imread(find_image_path(stem), cv2.IMREAD_GRAYSCALE)
    fov = field_of_view_mask(gray, erode_ksize=FOV_ERODE_KSIZE)
    return float(gray[fov > 0].mean())


def load_per_image_metrics(path):
    return {r["id"]: r for r in json.load(open(path))["per_image"]}


def summarize(rows, key):
    """Moyenne/écart-type/n par valeur de `key`, en gardant l'ordre d'apparition."""
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)

    summary = []
    for label, group in groups.items():
        summary.append({
            "category": label,
            "n": len(group),
            "mean_dice_unet": float(np.mean([r["dice_unet"] for r in group])),
            "std_dice_unet": float(np.std([r["dice_unet"] for r in group])),
            "mean_iou_unet": float(np.mean([r["iou_unet"] for r in group])),
            "mean_area_error_pct_unet": float(np.mean([r["area_error_pct_unet"] for r in group])),
            "mean_dice_baseline": float(np.mean([r["dice_baseline"] for r in group])),
            "mean_area_error_pct_baseline": float(np.mean([r["area_error_pct_baseline"] for r in group])),
        })
    return summary


def print_summary_table(title, summary):
    print(f"\n{title} (n = nombre d'images de test dans la catégorie)")
    header = f"{'Catégorie':<22}{'n':>4}{'Dice U-Net':>13}{'IoU U-Net':>12}{'Err. surf. U-Net':>18}{'Dice baseline':>16}"
    print(header)
    for row in summary:
        print(f"{row['category']:<22}{row['n']:>4}{row['mean_dice_unet']:>13.3f}"
              f"{row['mean_iou_unet']:>12.3f}{row['mean_area_error_pct_unet']:>17.1f}%"
              f"{row['mean_dice_baseline']:>16.3f}")
        if row["n"] < 4:
            print(f"  (n={row['n']} : moyenne indicative seulement, pas de portée statistique)")


def main():
    area_frac = load_area_fractions()
    unet = load_per_image_metrics(os.path.join(PRED_DIR, "test_metrics.json"))
    baseline = load_per_image_metrics(os.path.join(PRED_DIR, "test_baseline_metrics.json"))

    dataset_ids = list(area_frac.keys())
    brightness_by_id = {stem: field_brightness(stem) for stem in dataset_ids}
    brightness_values = np.array(list(brightness_by_id.values()))
    brightness_mean = brightness_values.mean()
    brightness_std = brightness_values.std()

    test_ids = load_ids(os.path.join(SPLITS_DIR, "test.txt"))

    rows = []
    for stem in test_ids:
        area_pct = area_frac[stem] * 100
        z = (brightness_by_id[stem] - brightness_mean) / brightness_std
        rows.append({
            "id": stem,
            "area_fraction_pct": area_pct,
            "surface_category": "quasi fermée (< 3 %)" if area_pct < AREA_THRESHOLD_PCT else "normale (>= 3 %)",
            "brightness": brightness_by_id[stem],
            "brightness_z": z,
            "lighting_category": "atypique" if abs(z) > BRIGHTNESS_Z_THRESHOLD else "normale",
            "dice_unet": unet[stem]["dice"],
            "iou_unet": unet[stem]["iou"],
            "area_error_pct_unet": unet[stem]["area_error_pct"],
            "dice_baseline": baseline[stem]["dice"],
            "area_error_pct_baseline": baseline[stem]["area_error_pct"],
        })

    os.makedirs(PRED_DIR, exist_ok=True)
    with open(OUT_PER_IMAGE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Détail par image -> {OUT_PER_IMAGE}")

    surface_summary = summarize(rows, "surface_category")
    lighting_summary = summarize(rows, "lighting_category")
    print_summary_table("Par surface relative de la plaie", surface_summary)
    print_summary_table("Par éclairage du champ (z-score de luminosité vs les 97 images du dataset)", lighting_summary)

    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "category", "n", "mean_dice_unet", "std_dice_unet",
                                                "mean_iou_unet", "mean_area_error_pct_unet",
                                                "mean_dice_baseline", "mean_area_error_pct_baseline"])
        writer.writeheader()
        for row in surface_summary:
            writer.writerow({"group": "surface", **row})
        for row in lighting_summary:
            writer.writerow({"group": "lighting", **row})
    print(f"\nRésumé par catégorie -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
