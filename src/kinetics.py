"""Suivi temporel de la cicatrisation : calcule la surface de la plaie sur une
série d'images du même échantillon prises à différents temps (0h, 24h, 48h...)
et le % de fermeture par rapport à l'image de référence (temps le plus faible).

Usage en ligne de commande:
    python -m src.kinetics --images data/kinetics/0h.jpg data/kinetics/24h.jpg data/kinetics/48h.jpg
    python -m src.kinetics --images data/kinetics/*.jpg --times 0 24 48 --pixels-per-cm 37.8

Les fonctions compute_kinetics_from_images / save_csv / build_figure / save_plot
sont aussi importées par app.py pour l'onglet "Suivi de cicatrisation", afin de
ne pas dupliquer la logique entre le script CLI et l'interface Streamlit.
"""
import argparse
import csv
import os
import re
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
from measure import area_px_to_cm2, wound_area_px
from predict import load_model, predict_mask

OUT_DIR = "outputs/kinetics"


def guess_time_h(filename: str) -> float:
    """Extrait un temps en heures depuis le nom de fichier (ex: '24h.jpg' -> 24)."""
    match = re.search(r"(\d+)\s*h", os.path.basename(filename), re.IGNORECASE)
    if not match:
        raise ValueError(f"Impossible de déduire le temps depuis le nom '{filename}'; précisez-le manuellement.")
    return float(match.group(1))


def compute_kinetics_from_images(model, img_size, device, images_with_times, threshold=0.5, pixels_per_cm=None):
    """images_with_times: liste de (nom, image_bgr, temps_h) déjà chargées en mémoire.

    Prédit la surface de la plaie sur chaque image et calcule le % de fermeture
    par rapport à l'image de référence (temps le plus faible, généralement 0h).

    Retourne (rows, masks_by_image) :
    - rows: liste de dict triée par temps croissant
      [{"time_h", "image", "area_px2", "area_cm2" (si pixels_per_cm), "closure_pct"}, ...]
    - masks_by_image: dict {nom_image: masque binaire prédit}, réutilisable pour
      tracer le contour détecté sans refaire l'inférence.
    """
    rows = []
    masks_by_image = {}
    for name, image_bgr, t in images_with_times:
        mask = predict_mask(model, image_bgr, img_size, device, threshold=threshold)
        masks_by_image[name] = mask
        area_px = wound_area_px(mask)

        row = {"time_h": t, "image": name, "area_px2": area_px}
        if pixels_per_cm:
            row["area_cm2"] = round(area_px_to_cm2(area_px, pixels_per_cm), 2)
        rows.append(row)

    rows.sort(key=lambda r: r["time_h"])

    area_t0 = rows[0]["area_px2"]
    for row in rows:
        row["closure_pct"] = round(100 * (area_t0 - row["area_px2"]) / area_t0, 1) if area_t0 > 0 else 0.0

    return rows, masks_by_image


def compute_kinetics(image_paths, times_h, run_name, threshold=0.5, pixels_per_cm=None, device=None):
    """Wrapper pour le script CLI : charge le modèle et les images depuis le disque."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = load_model(run_name, device)

    images_with_times = []
    for path, t in zip(image_paths, times_h):
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image illisible: {path}")
        images_with_times.append((os.path.basename(path), image_bgr, t))

    rows, _masks_by_image = compute_kinetics_from_images(
        model, img_size, device, images_with_times, threshold, pixels_per_cm
    )
    return rows


def save_csv(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "kinetics.csv")
    fieldnames = ["time_h", "image", "area_px2"]
    if "area_cm2" in rows[0]:
        fieldnames.append("area_cm2")
    fieldnames.append("closure_pct")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_figure(rows):
    times = [r["time_h"] for r in rows]
    closures = [r["closure_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, closures, marker="o")
    ax.set_xlabel("Temps (h)")
    ax.set_ylabel("Fermeture de la plaie (%)")
    ax.set_title(f"Évolution de la fermeture de la plaie ({len(rows)} temps de mesure)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_plot(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "kinetics_curve.png")
    fig = build_figure(rows)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Suivi temporel de la cicatrisation (% de fermeture par rapport à t0).")
    parser.add_argument("--images", nargs="+", required=True, help="Chemins des images, dans n'importe quel ordre")
    parser.add_argument("--times", nargs="+", type=float, default=None,
                         help="Temps en heures pour chaque image (même ordre que --images). "
                              "Si omis, déduit du nom de fichier (ex: '24h.jpg' -> 24)")
    parser.add_argument("--run_name", type=str, default="unet_resnet34")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pixels-per-cm", type=float, default=None,
                         help="Ratio pixels/cm si un objet de référence de taille connue est visible")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    times_h = args.times if args.times is not None else [guess_time_h(p) for p in args.images]
    if len(times_h) != len(args.images):
        raise ValueError("--times doit avoir le même nombre de valeurs que --images")

    rows = compute_kinetics(
        args.images, times_h, args.run_name,
        threshold=args.threshold, pixels_per_cm=args.pixels_per_cm,
    )

    csv_path = save_csv(rows, args.out_dir)
    plot_path = save_plot(rows, args.out_dir)

    print(f"{'Temps (h)':>10} {'Surface (px²)':>15} {'% fermeture':>12}")
    for row in rows:
        print(f"{row['time_h']:>10.1f} {row['area_px2']:>15} {row['closure_pct']:>12.1f}")

    print(f"\nCSV    -> {csv_path}")
    print(f"Courbe -> {plot_path}")


if __name__ == "__main__":
    main()
