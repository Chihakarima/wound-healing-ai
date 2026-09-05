"""Évaluation du modèle entraîné (Dice, IoU, precision, recall, Hausdorff,
erreur de surface) + visuels de comparaison prédiction / vérité terrain.

Évalue en résolution native : le masque prédit (calculé sur l'image
redimensionnée en img_size x img_size, comme le voit le réseau) est
réupsamplé à la résolution originale avant d'être comparé au masque de
vérité terrain original (comme le fait predict.py / l'app Streamlit). Ça
correspond à ce qu'un utilisateur voit réellement, et ça rend les métriques
comparables à celles de la baseline classique (src/baseline.py), qui
travaille elle aussi en résolution native. Évaluer sur le masque
redimensionné à img_size (comme le faisait une version précédente de ce
script) donne un Dice quasi identique mais un Hausdorff non comparable à la
baseline : l'écart de discrétisation entre 384px et la résolution native
(jusqu'à ~1600px) domine la distance mesurée.

Usage:
    python -m src.evaluate [--run_name unet_resnet34] [--split test]
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
from measure import hausdorff_distance_px, normalize_hausdorff, wound_area_px
from metrics import confusion_counts, dice_score, iou_score, precision_score, recall_score
from predict import load_model, predict_mask

SPLITS_DIR = "data/splits"
MASKS_DIR = "data/masks_binary"
PRED_DIR = "outputs/predictions"


def make_comparison_image(image_bgr: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    gt = (gt_mask * 255).astype(np.uint8)
    pred = (pred_mask * 255).astype(np.uint8)

    overlay = image_bgr.copy()
    gt_contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)    # vert = vérité terrain
    cv2.drawContours(overlay, pred_contours, -1, (0, 0, 255), 2)  # rouge = prédiction

    gt_bgr = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    return np.hstack([image_bgr, gt_bgr, pred_bgr, overlay])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="unet_resnet34")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                         help="'test' est un jeu tenu à l'écart de l'entraînement et de la "
                              "sélection du checkpoint (contrairement à 'val'), donc seul son "
                              "score est une vraie estimation de généralisation.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = load_model(args.run_name, device)

    ids = load_ids(os.path.join(SPLITS_DIR, f"{args.split}.txt"))
    out_dir = os.path.join(PRED_DIR, f"{args.split}_comparisons")
    os.makedirs(out_dir, exist_ok=True)

    dices, ious, precisions, recalls = [], [], [], []
    hausdorffs, hausdorffs_norm = [], []
    area_errors_px, area_errors_pct = [], []
    per_image = []

    for stem in ids:
        image_bgr = cv2.imread(find_image_path(stem))
        gt_mask = cv2.imread(os.path.join(MASKS_DIR, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        h = min(image_bgr.shape[0], gt_mask.shape[0])
        w = min(image_bgr.shape[1], gt_mask.shape[1])
        image_bgr, gt_mask = image_bgr[:h, :w], gt_mask[:h, :w]
        gt_mask = (gt_mask > 127).astype(np.uint8)

        pred_mask = predict_mask(model, image_bgr, img_size, device, threshold=args.threshold)

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
        area_error_px = abs(pred_area - gt_area)
        area_error_pct = 100 * area_error_px / max(gt_area, 1)

        dices.append(dice)
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        area_errors_px.append(area_error_px)
        area_errors_pct.append(area_error_pct)
        if hd is not None:
            hausdorffs.append(hd)
            hausdorffs_norm.append(hd_norm)

        per_image.append({
            "id": stem, "dice": dice, "iou": iou, "precision": precision, "recall": recall,
            "gt_area_px": gt_area, "pred_area_px": pred_area,
            "area_error_px": area_error_px, "area_error_pct": area_error_pct,
            "hausdorff_px": hd, "hausdorff_normalized": hd_norm,
        })

        comp = make_comparison_image(image_bgr, gt_mask, pred_mask)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_comparison.png"), comp)

    def _report(name, values):
        print(f"{args.split.capitalize()} {name} moyen: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

    _report("Dice", dices)
    _report("IoU", ious)
    _report("Precision", precisions)
    _report("Recall", recalls)
    if hausdorffs:
        _report("Hausdorff (px, résolution native)", hausdorffs)
        _report("Hausdorff normalisé (fraction de la diagonale)", hausdorffs_norm)
    if len(hausdorffs) < len(ids):
        print(f"  (Hausdorff non défini pour {len(ids) - len(hausdorffs)} image(s) sans contour prédit ou vérité terrain vide)")

    mae_area_px = float(np.mean(area_errors_px))
    rmse_area_px = float(np.sqrt(np.mean(np.square(area_errors_px))))
    mean_area_error_pct = float(np.mean(area_errors_pct))
    print(f"{args.split.capitalize()} erreur de surface : MAE={mae_area_px:.0f} px², "
          f"RMSE={rmse_area_px:.0f} px², erreur relative moyenne={mean_area_error_pct:.1f}%")
    print(f"Visuels de comparaison -> {out_dir}")

    with open(os.path.join(PRED_DIR, f"{args.split}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "resolution": "native",
            "mean_dice": float(np.mean(dices)), "std_dice": float(np.std(dices)),
            "mean_iou": float(np.mean(ious)), "std_iou": float(np.std(ious)),
            "mean_precision": float(np.mean(precisions)), "std_precision": float(np.std(precisions)),
            "mean_recall": float(np.mean(recalls)), "std_recall": float(np.std(recalls)),
            "mean_hausdorff_px": float(np.mean(hausdorffs)) if hausdorffs else None,
            "std_hausdorff_px": float(np.std(hausdorffs)) if hausdorffs else None,
            "mean_hausdorff_normalized": float(np.mean(hausdorffs_norm)) if hausdorffs_norm else None,
            "std_hausdorff_normalized": float(np.std(hausdorffs_norm)) if hausdorffs_norm else None,
            "mae_area_px2": mae_area_px,
            "rmse_area_px2": rmse_area_px,
            "mean_area_error_pct": mean_area_error_pct,
            "std_area_error_pct": float(np.std(area_errors_pct)),
            "per_image": per_image,
        }, f, indent=2)


if __name__ == "__main__":
    main()
