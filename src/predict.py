"""Prédiction sur une nouvelle image : masque, contour, surface (px^2 et cm^2 optionnel).

Usage:
    python -m src.predict --image chemin/vers/image.jpg [--pixels-per-cm 37.8]
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(__file__))
from measure import area_px_to_cm2, extract_contours, wound_area_px
from model import build_model

CKPT_DIR = "outputs/checkpoints"
PRED_DIR = "outputs/predictions"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(run_name: str, device: torch.device):
    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}_best.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(encoder_name=checkpoint["encoder"], encoder_weights=None).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["img_size"]


def predict_mask(model, image_bgr: np.ndarray, img_size: int, device: torch.device, threshold: float = 0.5) -> np.ndarray:
    """Retourne un masque binaire (0/1) à la résolution originale de l'image."""
    h0, w0 = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    image_t = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_t)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float().squeeze().cpu().numpy()

    pred_full = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return (pred_full > 0.5).astype(np.uint8)


def draw_contour_overlay(image_bgr: np.ndarray, contours) -> np.ndarray:
    """contours : un contour unique (array) ou une liste de contours (ex: extract_contours).
    Tous les contours fournis sont dessinés, pour que le tracé corresponde au masque complet."""
    if isinstance(contours, np.ndarray):
        contours = [contours]
    overlay = image_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Prédit le masque, le contour et la surface d'une plaie.")
    parser.add_argument("--image", required=True, help="Chemin de l'image à analyser")
    parser.add_argument("--run_name", type=str, default="unet_resnet34")
    parser.add_argument("--pixels-per-cm", type=float, default=None,
                         help="Ratio pixels/cm si un objet de référence de taille connue est visible")
    parser.add_argument("--out-dir", default=None, help="Dossier de sortie (par défaut outputs/predictions)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = load_model(args.run_name, device)

    image_path = Path(args.image)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image illisible: {image_path}")

    mask = predict_mask(model, image_bgr, img_size, device)
    contours = extract_contours(mask)

    area_px = wound_area_px(mask)
    print(f"Surface de la plaie: {area_px} pixels²")

    if args.pixels_per_cm:
        area_cm2 = area_px_to_cm2(area_px, args.pixels_per_cm)
        print(f"Surface de la plaie: {area_cm2:.2f} cm² (ratio {args.pixels_per_cm} px/cm)")
    else:
        print("Astuce: passez --pixels-per-cm pour obtenir la surface en cm² "
              "(ratio = longueur en pixels d'un objet de référence / sa taille réelle en cm).")

    out_dir = Path(args.out_dir) if args.out_dir else Path(PRED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), mask * 255)

    if contours:
        overlay = draw_contour_overlay(image_bgr, contours)
        overlay_path = out_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        print(f"Overlay -> {overlay_path}")

    print(f"Masque  -> {mask_path}")


if __name__ == "__main__":
    main()
