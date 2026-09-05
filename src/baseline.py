"""Baseline classique (sans IA) pour la segmentation de plaie, à comparer au U-Net.

La zone de plaie (gap) n'est pas caractérisée par une couleur ou une luminosité
différente du reste du champ (rose clair des deux côtés) : ce qui la distingue,
c'est l'absence de texture cellulaire (le gap est lisse, les zones couvertes de
cellules sont grainées). Un simple seuillage Otsu sur l'intensité brute ne
sépare donc pas gap et cellules ; on seuille à la place une carte de texture
locale (écart-type dans une fenêtre glissante), une approche comparable à
celle des outils classiques de mesure de scratch assay (ex: plugin ImageJ
"Wound Healing Size Tool").

Usage:
    python -m src.baseline [--split test]
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

SPLITS_DIR = "data/splits"
MASKS_DIR = "data/masks_binary"
PRED_DIR = "outputs/predictions"

TEXTURE_KSIZE = 25    # fenêtre (px, impair) du calcul de texture locale
FOV_ERODE_KSIZE = 41  # marge retirée sur le bord du champ (vignettage : contraste
                      # naturellement plus faible sur le pourtour, confondu sinon avec le gap)
MIN_BLOB_AREA = 200   # supprime les petits blobs de bruit après seuillage


def field_of_view_mask(gray, black_threshold=15, erode_ksize=0):
    """Isole le champ circulaire du microscope : le fond noir autour doit être
    exclu du seuillage, sinon Otsu sépare fond noir / champ clair au lieu de
    gap / cellules à l'intérieur du champ.

    erode_ksize > 0 : retire en plus une marge sur le pourtour intérieur du
    champ, où le vignettage (moins de lumière/contraste en bord d'objectif)
    fait chuter la texture locale indépendamment de la présence de cellules,
    ce qui sinon se confond avec le gap et fait fuir le masque prédit vers
    le bord (faux positifs)."""
    fov = (gray > black_threshold).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fov, connectivity=8)
    if n <= 1:
        return fov
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fov = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(fov)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    if erode_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
        filled = cv2.erode(filled, kernel)
    return filled


def local_texture(gray, ksize=TEXTURE_KSIZE):
    """Écart-type local (fenêtre ksize x ksize) : élevé sur les zones couvertes
    de cellules (texture grainée), faible sur le gap (zone lisse)."""
    gray = gray.astype(np.float32)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(ksize, ksize))
    mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(ksize, ksize))
    variance = np.clip(mean_sq - mean * mean, 0, None)
    return np.sqrt(variance)


def predict_mask_classical(image_bgr):
    """Segmentation classique du gap : seuillage Otsu sur la carte de texture
    locale (faible texture = gap), restreint au champ du microscope (marge de
    vignettage exclue)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    fov = field_of_view_mask(gray, erode_ksize=FOV_ERODE_KSIZE)

    texture = local_texture(gray)
    # normalize() sur les pixels du champ uniquement : sinon le grand fond noir
    # (texture ~0, largement majoritaire hors du champ) écrase l'échelle et les
    # variations de texture pertinentes à l'intérieur du champ sont compressées
    # dans une toute petite plage de valeurs.
    texture_u8 = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX, mask=fov).astype(np.uint8)

    # THRESH_BINARY_INV : on garde les pixels EN DESSOUS du seuil Otsu (faible texture -> gap)
    _, low_texture = cv2.threshold(texture_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    low_texture = cv2.bitwise_and(low_texture, fov)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(low_texture, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    mask = np.zeros_like(cleaned)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA:
            mask[labels == i] = 1
    return mask.astype(np.uint8)


def make_comparison_image(image_bgr, gt_mask, pred_mask):
    gt = (gt_mask * 255).astype(np.uint8)
    pred = (pred_mask * 255).astype(np.uint8)
    overlay = image_bgr.copy()
    gt_contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)    # vert = vérité terrain
    cv2.drawContours(overlay, pred_contours, -1, (0, 0, 255), 2)  # rouge = prédiction baseline
    gt_bgr = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    return np.hstack([image_bgr, gt_bgr, pred_bgr, overlay])


def main():
    parser = argparse.ArgumentParser(description="Baseline classique (texture + Otsu) évaluée sur un split.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    ids = load_ids(os.path.join(SPLITS_DIR, f"{args.split}.txt"))
    out_dir = os.path.join(PRED_DIR, f"{args.split}_baseline_comparisons")
    os.makedirs(out_dir, exist_ok=True)

    dices, ious, precisions, recalls, hausdorffs, hausdorffs_norm = [], [], [], [], [], []
    area_errors_px, area_errors = [], []
    per_image = []
    for stem in ids:
        image_bgr = cv2.imread(find_image_path(stem))
        gt_mask = cv2.imread(os.path.join(MASKS_DIR, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        h = min(image_bgr.shape[0], gt_mask.shape[0])
        w = min(image_bgr.shape[1], gt_mask.shape[1])
        image_bgr, gt_mask = image_bgr[:h, :w], gt_mask[:h, :w]
        gt_mask = (gt_mask > 127).astype(np.uint8)

        pred_mask = predict_mask_classical(image_bgr)

        pred_t = torch.from_numpy(pred_mask.astype(bool)).unsqueeze(0)
        gt_t = torch.from_numpy(gt_mask.astype(bool)).unsqueeze(0)
        tp, fp, fn, _ = confusion_counts(pred_t, gt_t)
        dice = dice_score(tp, fp, fn).item()
        iou = iou_score(tp, fp, fn).item()
        precision = precision_score(tp, fp).item()
        recall = recall_score(tp, fn).item()
        hd = hausdorff_distance_px(pred_mask, gt_mask)

        gt_area = wound_area_px(gt_mask)
        pred_area = wound_area_px(pred_mask)
        area_error_px = abs(pred_area - gt_area)
        area_error_pct = 100 * area_error_px / max(gt_area, 1)

        dices.append(dice)
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        area_errors_px.append(area_error_px)
        area_errors.append(area_error_pct)
        hd_norm = None
        if hd is not None:
            hausdorffs.append(hd)
            hd_norm = normalize_hausdorff(hd, gt_mask.shape)
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
    _report("Erreur de surface (%)", area_errors)
    if hausdorffs:
        _report("Hausdorff (px, résolution native)", hausdorffs)
        _report("Hausdorff normalisé (fraction de la diagonale)", hausdorffs_norm)
    if len(hausdorffs) < len(ids):
        print(f"  (Hausdorff non défini pour {len(ids) - len(hausdorffs)} image(s) sans contour prédit ou vérité terrain vide)")

    mae_area_px = float(np.mean(area_errors_px))
    rmse_area_px = float(np.sqrt(np.mean(np.square(area_errors_px))))
    print(f"{args.split.capitalize()} erreur de surface : MAE={mae_area_px:.0f} px², "
          f"RMSE={rmse_area_px:.0f} px², erreur relative moyenne={np.mean(area_errors):.1f}%")
    print(f"Visuels de comparaison -> {out_dir}")

    with open(os.path.join(PRED_DIR, f"{args.split}_baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "method": "otsu_texture",
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
            "mean_area_error_pct": float(np.mean(area_errors)), "std_area_error_pct": float(np.std(area_errors)),
            "per_image": per_image,
        }, f, indent=2)


if __name__ == "__main__":
    main()
