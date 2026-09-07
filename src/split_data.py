"""Create a reproducible train/val/test split over the prepared binary masks.

test.txt is a held-out set never used for training or for early-stopping model
selection (unlike val.txt), so metrics computed on it are a genuine estimate of
generalization rather than a number the checkpoint was implicitly picked against."""
import argparse
import glob
import os
import random

MASKS_DIR = "data/masks_binary"
SPLITS_DIR = "data/splits"
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

# Flagged during mask preparation (outputs/qc_masks/): traced outline only
# covers part of the gap (top portion undetected / not drawn). Exclude until
# manually corrected -- see outputs/qc_masks/74.jpg.
EXCLUDE_IDS = {"74"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED,
                         help="graine du split (défaut: celle des résultats canoniques du README)")
    parser.add_argument("--out_dir", type=str, default=SPLITS_DIR,
                         help="dossier de sortie des 3 fichiers de split (défaut: data/splits, "
                              "les splits canoniques -- utiliser un autre dossier pour une étude "
                              "multi-seed sans écraser ces fichiers)")
    args = parser.parse_args()

    ids = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(MASKS_DIR, "*.png"))
    )
    ids = [i for i in ids if i not in EXCLUDE_IDS]

    rng = random.Random(args.seed)
    rng.shuffle(ids)

    n_val = max(1, round(len(ids) * VAL_FRACTION))
    n_test = max(1, round(len(ids) * TEST_FRACTION))
    val_ids = sorted(ids[:n_val], key=int)
    test_ids = sorted(ids[n_val:n_val + n_test], key=int)
    train_ids = sorted(ids[n_val + n_test:], key=int)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        with open(os.path.join(args.out_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(split_ids) + "\n")

    print(f"train: {len(train_ids)} images, val: {len(val_ids)} images, test: {len(test_ids)} images")
    print(f"excluded: {sorted(EXCLUDE_IDS)}")


if __name__ == "__main__":
    main()
