"""Évaluation du modèle entraîné sur le jeu de validation (Dice, IoU) + visuels
de comparaison prédiction / vérité terrain.

Usage:
    python -m src.evaluate [--run_name unet_resnet34] [--img_size 384]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import PlateSegmentationDataset, load_ids
from measure import hausdorff_distance_px
from metrics import confusion_counts, dice_score, iou_score, precision_score, recall_score
from model import build_model

SPLITS_DIR = "data/splits"
CKPT_DIR = "outputs/checkpoints"
PRED_DIR = "outputs/predictions"

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denormalize(image_t: torch.Tensor) -> np.ndarray:
    img = image_t.cpu().numpy().transpose(1, 2, 0)
    img = (img * STD + MEAN) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def make_comparison_image(image_t, gt_mask_t, pred_mask: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(denormalize(image_t), cv2.COLOR_RGB2BGR)
    gt = (gt_mask_t.squeeze().cpu().numpy() * 255).astype(np.uint8)
    pred = (pred_mask * 255).astype(np.uint8)

    overlay = img.copy()
    gt_contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)    # vert = vérité terrain
    cv2.drawContours(overlay, pred_contours, -1, (0, 0, 255), 2)  # rouge = prédiction

    gt_bgr = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    return np.hstack([img, gt_bgr, pred_bgr, overlay])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="unet_resnet34")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                         help="'test' est un jeu tenu à l'écart de l'entraînement et de la "
                              "sélection du checkpoint (contrairement à 'val'), donc seul son "
                              "score est une vraie estimation de généralisation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ids = load_ids(os.path.join(SPLITS_DIR, f"{args.split}.txt"))
    ds = PlateSegmentationDataset(ids, args.img_size, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    ckpt_path = os.path.join(CKPT_DIR, f"{args.run_name}_best.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(encoder_name=checkpoint["encoder"], encoder_weights=None).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    out_dir = os.path.join(PRED_DIR, f"{args.split}_comparisons")
    os.makedirs(out_dir, exist_ok=True)

    dices, ious, precisions, recalls, hausdorffs = [], [], [], [], []
    with torch.no_grad():
        for images, masks, ids_batch in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)

            preds = (torch.sigmoid(logits) > 0.5)
            tp, fp, fn, _ = confusion_counts(preds, masks.bool())
            dices.append(dice_score(tp, fp, fn).item())
            ious.append(iou_score(tp, fp, fn).item())
            precisions.append(precision_score(tp, fp).item())
            recalls.append(recall_score(tp, fn).item())

            pred_mask = preds.float().squeeze().cpu().numpy()
            gt_mask = masks[0].squeeze().cpu().numpy()
            hd = hausdorff_distance_px(pred_mask, gt_mask)
            if hd is not None:
                hausdorffs.append(hd)

            comp = make_comparison_image(images[0], masks[0], pred_mask)
            cv2.imwrite(os.path.join(out_dir, f"{ids_batch[0]}_comparison.png"), comp)

    def _report(name, values):
        print(f"{args.split.capitalize()} {name} moyen: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

    _report("Dice", dices)
    _report("IoU", ious)
    _report("Precision", precisions)
    _report("Recall", recalls)
    if hausdorffs:
        _report("Hausdorff (px)", hausdorffs)
    if len(hausdorffs) < len(ids):
        print(f"  (Hausdorff non défini pour {len(ids) - len(hausdorffs)} image(s) sans contour prédit ou vérité terrain vide)")
    print(f"Visuels de comparaison -> {out_dir}")

    with open(os.path.join(PRED_DIR, f"{args.split}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "mean_dice": float(np.mean(dices)), "std_dice": float(np.std(dices)),
            "mean_iou": float(np.mean(ious)), "std_iou": float(np.std(ious)),
            "mean_precision": float(np.mean(precisions)), "std_precision": float(np.std(precisions)),
            "mean_recall": float(np.mean(recalls)), "std_recall": float(np.std(recalls)),
            "mean_hausdorff_px": float(np.mean(hausdorffs)) if hausdorffs else None,
            "std_hausdorff_px": float(np.std(hausdorffs)) if hausdorffs else None,
            "per_image": [
                {"id": i, "dice": d, "iou": u, "precision": p, "recall": r}
                for i, d, u, p, r in zip(ids, dices, ious, precisions, recalls)
            ],
        }, f, indent=2)


if __name__ == "__main__":
    main()
