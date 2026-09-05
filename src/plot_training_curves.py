"""Courbes d'apprentissage (train vs validation) à partir du log CSV produit
par train.py. Pour le rapport : montre que le modèle apprend, permet de
repérer un éventuel surapprentissage, et justifie la sélection du checkpoint
sur le Dice de validation.

Volontairement train + validation seulement, pas test : le jeu de test doit
rester un jeu tenu à l'écart évalué une seule fois (voir evaluate.py), pas une
courbe qu'on regarde évoluer pendant l'entraînement.

Usage:
    python -m src.plot_training_curves [--run_name unet_resnet34_holdout]
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "outputs"
FIG_DIR = "outputs/figures"


def load_log(run_name):
    path = os.path.join(LOG_DIR, f"{run_name}_training_log.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in row:
            if key not in ("encoder_frozen",):
                row[key] = float(row[key])
    return rows


def unfreeze_epoch(rows):
    """Première époque où l'encodeur est dégelé (fin du warmup), pour repérer
    sur les courbes le changement de régime (LR divisé par 10, plus de
    paramètres entraînés) qui produit un saut/coude visible."""
    for row in rows:
        if row["encoder_frozen"] == "False":
            return int(row["epoch"])
    return None


def plot_loss(rows, out_dir):
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["train_loss"] for r in rows], label="Train", color="#0F766E")
    ax.plot(epochs, [r["val_loss"] for r in rows], label="Validation", color="#DC2626")

    unfreeze = unfreeze_epoch(rows)
    if unfreeze:
        ax.axvline(unfreeze, color="#94A3B8", linestyle="--", linewidth=1)
        ax.annotate("dégel de l'encodeur", (unfreeze, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(4, -12), fontsize=8, color="#64748B")

    ax.set_xlabel("Époque")
    ax.set_ylabel("Loss (Dice + BCE)")
    ax.set_title("Figure — Évolution de la loss (train / validation)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "figure3_loss_curves.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_dice(rows, out_dir):
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["train_dice"] for r in rows], label="Train", color="#0F766E")
    ax.plot(epochs, [r["val_dice"] for r in rows], label="Validation", color="#DC2626")

    best_epoch = max(rows, key=lambda r: r["val_dice"])
    ax.scatter([best_epoch["epoch"]], [best_epoch["val_dice"]], color="#DC2626", zorder=5, s=60)
    ax.annotate(f"meilleur checkpoint\n(époque {int(best_epoch['epoch'])}, Dice={best_epoch['val_dice']:.3f})",
                (best_epoch["epoch"], best_epoch["val_dice"]),
                textcoords="offset points", xytext=(8, -22), fontsize=8, color="#991B1B")

    unfreeze = unfreeze_epoch(rows)
    if unfreeze:
        ax.axvline(unfreeze, color="#94A3B8", linestyle="--", linewidth=1)
        ax.annotate("dégel de l'encodeur", (unfreeze, ax.get_ylim()[0]),
                    textcoords="offset points", xytext=(4, 6), fontsize=8, color="#64748B")

    ax.set_xlabel("Époque")
    ax.set_ylabel("Dice")
    ax.set_title("Figure — Évolution du Dice (train / validation)", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.text(0.5, 0.01,
             "Le Dice de validation dépasse celui de train : attendu ici, car train applique une\n"
             "augmentation de données forte (crop, flips, bruit...) qui rend la tâche plus difficile\n"
             "à ce stade, tandis que validation ne fait qu'un simple redimensionnement — ce n'est pas\n"
             "un signal de surapprentissage.",
             ha="center", fontsize=8, style="italic")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "figure4_dice_curves.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="unet_resnet34_holdout")
    args = parser.parse_args()

    rows = load_log(args.run_name)
    path_loss = plot_loss(rows, FIG_DIR)
    print(f"Figure loss -> {path_loss}")
    path_dice = plot_dice(rows, FIG_DIR)
    print(f"Figure Dice -> {path_dice}")


if __name__ == "__main__":
    main()
