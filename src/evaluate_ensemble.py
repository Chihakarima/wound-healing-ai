"""Évaluation de l'ensemble de modèles (src/ensemble.py) sur le jeu de test, dans les
mêmes conditions que src/evaluate.py (résolution native, mêmes métriques), pour vérifier
empiriquement si combiner plusieurs modèles améliore réellement Dice/IoU par rapport à un
seul modèle -- pas supposé, mesuré.

Calcule aussi la corrélation entre le % de zone incertaine (désaccord entre modèles) et le
Dice par image : si le signal de confiance est pertinent, les images à fort désaccord
doivent être celles où le modèle se trompe le plus.

Usage:
    python -m src.evaluate_ensemble [--split test]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import find_image_path, load_ids
from ensemble import RUN_NAMES_PAR_DEFAUT, load_ensemble, predict_ensemble
from measure import hausdorff_distance_px, normalize_hausdorff, wound_area_px
from metrics import confusion_counts, dice_score, iou_score, precision_score, recall_score

SPLITS_DIR = "data/splits"
MASKS_DIR = "data/masks_binary"
PRED_DIR = "outputs/predictions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_names", nargs="+", default=RUN_NAMES_PAR_DEFAUT)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_and_sizes = load_ensemble(args.run_names, device)

    ids = load_ids(os.path.join(SPLITS_DIR, f"{args.split}.txt"))

    dices, ious, precisions, recalls = [], [], [], []
    hausdorffs_norm = []
    area_errors_pct = []
    pct_incertain = []
    per_image = []

    for stem in ids:
        image_bgr = cv2.imread(find_image_path(stem))
        gt_mask = cv2.imread(os.path.join(MASKS_DIR, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        h = min(image_bgr.shape[0], gt_mask.shape[0])
        w = min(image_bgr.shape[1], gt_mask.shape[1])
        image_bgr, gt_mask = image_bgr[:h, :w], gt_mask[:h, :w]
        gt_mask = (gt_mask > 127).astype(np.uint8)

        resultat = predict_ensemble(models_and_sizes, image_bgr, device, threshold=args.threshold)
        pred_mask = resultat["mask_consensus"]

        pred_t = torch.from_numpy(pred_mask.astype(bool)).unsqueeze(0)
        gt_t = torch.from_numpy(gt_mask.astype(bool)).unsqueeze(0)
        tp, fp, fn, _ = confusion_counts(pred_t, gt_t)
        dice = dice_score(tp, fp, fn).item()
        iou = iou_score(tp, fp, fn).item()
        precision = precision_score(tp, fp).item()
        recall = recall_score(tp, fn).item()

        hd = hausdorff_distance_px(pred_mask, gt_mask)
        hd_norm = normalize_hausdorff(hd, gt_mask.shape) if hd is not None else None

        gt_area = wound_area_px(gt_mask)
        pred_area = wound_area_px(pred_mask)
        area_error_pct = 100 * abs(pred_area - gt_area) / max(gt_area, 1)

        dices.append(dice)
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        area_errors_pct.append(area_error_pct)
        pct_incertain.append(resultat["pct_zone_incertaine"])
        if hd_norm is not None:
            hausdorffs_norm.append(hd_norm)

        per_image.append({
            "id": stem, "dice": dice, "iou": iou,
            "area_error_pct": area_error_pct,
            "pct_zone_incertaine": resultat["pct_zone_incertaine"],
        })

    def _report(name, values):
        print(f"{args.split.capitalize()} {name} moyen (ensemble): {np.mean(values):.4f} (+/- {np.std(values):.4f})")

    _report("Dice", dices)
    _report("IoU", ious)
    _report("Precision", precisions)
    _report("Recall", recalls)
    if hausdorffs_norm:
        _report("Hausdorff normalisé", hausdorffs_norm)
    print(f"Erreur de surface relative moyenne : {np.mean(area_errors_pct):.1f}%")
    print(f"Zone incertaine moyenne : {np.mean(pct_incertain):.1f}% de la zone détectée")

    # Corrélation entre désaccord et Dice : un signal de confiance pertinent doit être
    # négativement corrélé au Dice (plus de désaccord -> Dice plus faible).
    if len(set(pct_incertain)) > 1:
        correlation = float(np.corrcoef(pct_incertain, dices)[0, 1])
        print(f"Corrélation (zone incertaine, Dice) : {correlation:.3f} "
              f"({'signal cohérent : plus de désaccord -> Dice plus faible' if correlation < 0 else 'signal non confirmé sur ce jeu de test'})")
    else:
        correlation = None

    with open(os.path.join(PRED_DIR, f"{args.split}_ensemble_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "run_names": args.run_names,
            "mean_dice": float(np.mean(dices)), "std_dice": float(np.std(dices)),
            "mean_iou": float(np.mean(ious)), "std_iou": float(np.std(ious)),
            "mean_precision": float(np.mean(precisions)), "std_precision": float(np.std(precisions)),
            "mean_recall": float(np.mean(recalls)), "std_recall": float(np.std(recalls)),
            "mean_hausdorff_normalized": float(np.mean(hausdorffs_norm)) if hausdorffs_norm else None,
            "mean_area_error_pct": float(np.mean(area_errors_pct)),
            "mean_pct_zone_incertaine": float(np.mean(pct_incertain)),
            "correlation_incertitude_dice": correlation,
            "per_image": per_image,
        }, f, indent=2)
    print(f"Résultats -> {PRED_DIR}/{args.split}_ensemble_metrics.json")


if __name__ == "__main__":
    main()
