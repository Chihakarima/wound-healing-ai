"""Ensemble de plusieurs modèles entraînés indépendamment (voir "Robustesse inter-seed"
dans le README : unet_resnet34_seed42/123/2026, mêmes hyperparamètres, split et
initialisation régénérés par graine) pour une prédiction plus robuste et une estimation
de la confiance par pixel : là où les modèles sont d'accord, la prédiction est fiable ;
là où ils divergent, c'est un signal qu'une vérification manuelle est utile -- typiquement
les extrémités fines de la plaie, déjà identifiées comme la source du Hausdorff instable.

Usage (script) :
    python -m src.ensemble --image chemin/vers/image.jpg
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from measure import area_px_to_cm2, extract_contours, wound_area_px
from predict import draw_contour_overlay, load_model, predict_mask

PRED_DIR = "outputs/predictions"

# Run canonique + les 3 graines de l'étude de robustesse (voir README, "Ingénierie") :
# 4 modèles indépendants au total, pas seulement les 3 graines, pour une estimation de
# confiance qui ne dépend pas d'un seul protocole d'entraînement.
RUN_NAMES_PAR_DEFAUT = [
    "unet_resnet34_holdout",
    "unet_resnet34_seed42",
    "unet_resnet34_seed123",
    "unet_resnet34_seed2026",
]


def load_ensemble(run_names: list[str], device: torch.device):
    """Charge plusieurs checkpoints. Retourne une liste de (model, img_size)."""
    return [load_model(run_name, device) for run_name in run_names]


def masque_consensus(masks: list[np.ndarray]) -> np.ndarray:
    """Masque binaire par vote majoritaire pixel par pixel : un pixel est "plaie" si
    strictement plus de la moitié des modèles le prédisent ainsi."""
    votes = np.sum(masks, axis=0)
    return (votes > len(masks) / 2).astype(np.uint8)


def carte_accord(masks: list[np.ndarray]) -> np.ndarray:
    """Carte du nombre de modèles (0..N) qui prédisent "plaie" à chaque pixel."""
    return np.sum(masks, axis=0).astype(np.uint8)


def pourcentage_zone_incertaine(masks: list[np.ndarray]) -> float:
    """% de la zone détectée (union des masques, pas l'image entière -- le fond où tous
    les modèles sont d'accord qu'il n'y a pas de plaie n'est pas une zone "incertaine"
    pertinente) où les modèles ne sont pas unanimes.

    0.0 si aucun modèle ne détecte de plaie (union vide) : rien à qualifier d'incertain
    dans ce cas plutôt qu'une division par zéro."""
    votes = np.sum(masks, axis=0)
    n = len(masks)
    union = votes > 0
    if not union.any():
        return 0.0
    desaccord = union & (votes < n)
    return float(100.0 * desaccord.sum() / union.sum())


def overlay_confiance(image_bgr: np.ndarray, carte: np.ndarray, n_modeles: int,
                       alpha: float = 0.45) -> np.ndarray:
    """Superpose en vert les zones où tous les modèles sont d'accord (plaie détectée par
    tous), et en rouge les zones de désaccord partiel -- le fond (aucun modèle ne détecte
    de plaie) n'est pas teinté."""
    overlay = image_bgr.astype(np.float32).copy()
    vert = np.array([0, 200, 0], dtype=np.float32)   # BGR
    rouge = np.array([0, 0, 220], dtype=np.float32)

    zone_accord = carte == n_modeles
    zone_desaccord = (carte > 0) & (carte < n_modeles)

    overlay[zone_accord] = (1 - alpha) * overlay[zone_accord] + alpha * vert
    overlay[zone_desaccord] = (1 - alpha) * overlay[zone_desaccord] + alpha * rouge
    return overlay.astype(np.uint8)


def predict_ensemble(models_and_sizes, image_bgr: np.ndarray, device: torch.device,
                      threshold: float = 0.5) -> dict:
    """Prédit avec chaque modèle de l'ensemble (models_and_sizes: sortie de
    load_ensemble()), puis combine.

    Retourne {"masks", "mask_consensus", "carte_accord", "pct_zone_incertaine"}.
    """
    masks = [
        predict_mask(model, image_bgr, img_size, device, threshold=threshold)
        for model, img_size in models_and_sizes
    ]
    return {
        "masks": masks,
        "mask_consensus": masque_consensus(masks),
        "carte_accord": carte_accord(masks),
        "pct_zone_incertaine": pourcentage_zone_incertaine(masks),
    }


def main():
    parser = argparse.ArgumentParser(description="Prédiction par ensemble de modèles + carte de confiance.")
    parser.add_argument("--image", required=True, help="Chemin de l'image à analyser")
    parser.add_argument("--run_names", nargs="+", default=RUN_NAMES_PAR_DEFAUT)
    parser.add_argument("--pixels-per-cm", type=float, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_and_sizes = load_ensemble(args.run_names, device)

    image_path = Path(args.image)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image illisible: {image_path}")

    resultat = predict_ensemble(models_and_sizes, image_bgr, device)
    mask = resultat["mask_consensus"]
    area_px = wound_area_px(mask)

    print(f"Ensemble de {len(args.run_names)} modèles : {', '.join(args.run_names)}")
    print(f"Surface de la plaie (consensus) : {area_px} pixels²")
    print(f"Zone incertaine (modèles en désaccord) : {resultat['pct_zone_incertaine']:.1f}% de la zone détectée")
    if args.pixels_per_cm:
        print(f"Surface : {area_px_to_cm2(area_px, args.pixels_per_cm):.2f} cm²")

    out_dir = Path(args.out_dir) if args.out_dir else Path(PRED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / f"{image_path.stem}_mask_ensemble.png"), mask * 255)
    contours = extract_contours(mask)
    if contours:
        overlay = draw_contour_overlay(image_bgr, contours)
        cv2.imwrite(str(out_dir / f"{image_path.stem}_overlay_ensemble.png"), overlay)
    confiance = overlay_confiance(image_bgr, resultat["carte_accord"], len(args.run_names))
    cv2.imwrite(str(out_dir / f"{image_path.stem}_confiance.png"), confiance)
    print(f"Sorties -> {out_dir}")


if __name__ == "__main__":
    main()
