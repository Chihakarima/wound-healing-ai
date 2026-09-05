"""Génère les deux figures d'analyse d'erreur du rapport :

Figure 1 : grille d'exemples (image originale | vérité terrain | U-Net | baseline)
pour 4 cas représentatifs (bonne segmentation, cas difficile, plaie très petite
[image 93], échec de la baseline classique).

Figure 2 : surface relative de la plaie (%) vs Dice du U-Net sur les 14 images
de test, pour visualiser la tendance observée sur les petites plaies (voir
docstring de plot_area_vs_dice : pas une relation statistique démontrée sur
un échantillon de cette taille).

Usage:
    python -m src.error_analysis_figures
"""
import csv
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline import predict_mask_classical
from dataset import find_image_path, load_ids
from measure import extract_contours, wound_area_px
from predict import load_model, predict_mask

SPLITS_DIR = "data/splits"
MASKS_DIR = "data/masks_binary"
PRED_DIR = "outputs/predictions"
QC_AREA_FRACTIONS = "outputs/qc_masks/_area_fractions.csv"
FIG_DIR = "outputs/figures"

# (id, étiquette de la ligne dans la figure 1)
EXAMPLE_CASES = [
    ("39", "Bonne segmentation (id 39)"),
    ("50", "Cas difficile (id 50)"),
    ("93", "Plaie très petite (id 93)"),
    ("76", "Échec de la baseline (id 76)"),
]


def load_image_and_gt(stem):
    image_bgr = cv2.imread(find_image_path(stem))
    gt_mask = cv2.imread(os.path.join(MASKS_DIR, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    h = min(image_bgr.shape[0], gt_mask.shape[0])
    w = min(image_bgr.shape[1], gt_mask.shape[1])
    return image_bgr[:h, :w], (gt_mask[:h, :w] > 127).astype(np.uint8)


def overlay_contour(image_bgr, mask, color_bgr):
    contours = extract_contours(mask)
    overlay = image_bgr.copy()
    if contours:
        cv2.drawContours(overlay, contours, -1, color_bgr, 3)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def plot_example_grid(model, img_size, device):
    n = len(EXAMPLE_CASES)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n))
    col_titles = ["Image originale", "Vérité terrain", "U-Net", "Baseline (texture+Otsu)"]

    for row, (stem, row_label) in enumerate(EXAMPLE_CASES):
        image_bgr, gt_mask = load_image_and_gt(stem)
        unet_mask = predict_mask(model, image_bgr, img_size, device, threshold=0.5)
        baseline_mask = predict_mask_classical(image_bgr)

        panels = [
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            overlay_contour(image_bgr, gt_mask, (0, 255, 0)),
            overlay_contour(image_bgr, unet_mask, (0, 0, 255)),
            overlay_contour(image_bgr, baseline_mask, (255, 128, 0)),
        ]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
        axes[row, 0].set_ylabel(row_label, fontsize=10, rotation=90)

    fig.suptitle("Figure 1 — Exemples représentatifs : vérité terrain (vert), U-Net (rouge), baseline (orange)",
                  fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "figure1_exemples.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_area_vs_dice():
    """Surface relative de la plaie (%) vs Dice du U-Net, sur les 14 images de
    test uniquement. Avec un échantillon aussi petit, une droite de régression
    donnerait une fausse impression de rigueur statistique : le graphique
    montre les points bruts + une tendance lissée à titre indicatif seulement,
    et la légende reste volontairement prudente ("tendance observée", pas
    "relation statistique démontrée")."""
    area_frac = {}
    with open(QC_AREA_FRACTIONS) as f:
        for row in csv.reader(f):
            if row[0] == "id":
                continue
            area_frac[row[0]] = float(row[1])

    unet = json.load(open(os.path.join(PRED_DIR, "test_metrics.json")))["per_image"]
    ids = [r["id"] for r in unet]
    x = np.array([area_frac[i] * 100 for i in ids])
    y = np.array([r["dice"] for r in unet])

    order = np.argsort(x)
    x_sorted, y_sorted, ids_sorted = x[order], y[order], [ids[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_sorted, y_sorted, color="#0F766E", zorder=3, s=60)
    for xi, yi, id_ in zip(x_sorted, y_sorted, ids_sorted):
        ax.annotate(id_, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Tendance lissée (moyenne mobile sur les points triés par surface) : sert
    # uniquement à guider l'oeil, pas une régression avec intervalle de confiance.
    if len(x_sorted) >= 3:
        window = 5
        trend = np.convolve(y_sorted, np.ones(window) / window, mode="valid")
        trend_x = x_sorted[window // 2: window // 2 + len(trend)]
        ax.plot(trend_x, trend, color="#94A3B8", linestyle="--", zorder=2,
                 label="tendance lissée (moyenne mobile, indicative)")
        ax.legend(fontsize=8, loc="lower right")

    ax.set_xlabel("Surface relative de la plaie (% du champ, vérité terrain)")
    ax.set_ylabel("Dice (U-Net, résolution native)")
    ax.set_title("Figure 2 — Surface de la plaie vs Dice (14 images de test)", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.text(0.5, -0.02,
             "Tendance observée sur 14 images : les performances semblent diminuer pour les plaies\n"
             "de très petite surface relative — ce n'est pas une relation statistique démontrée.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "figure2_surface_vs_dice.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = load_model("unet_resnet34_holdout", device)

    path1 = plot_example_grid(model, img_size, device)
    print(f"Figure 1 -> {path1}")

    path2 = plot_area_vs_dice()
    print(f"Figure 2 -> {path2}")


if __name__ == "__main__":
    main()
