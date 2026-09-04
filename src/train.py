"""Fine-tune a pretrained U-Net (ImageNet encoder) on the plate/wound
segmentation dataset. Tracks loss and Dice/IoU per epoch (train vs val) so
the learning curves can be plotted directly for the report."""
import argparse
import csv
import os
import sys
import time

import mlflow
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import PlateSegmentationDataset, load_ids
from metrics import DiceBCELoss, confusion_counts, dice_score, iou_score
from model import build_model

SPLITS_DIR = "data/splits"
CKPT_DIR = "outputs/checkpoints"
LOG_PATH = "outputs/training_log.csv"
MLFLOW_EXPERIMENT = "wound-segmentation"


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)

    total_loss, n_batches = 0.0, 0
    tp_sum = fp_sum = fn_sum = 0.0
    n_samples = 0

    with torch.set_grad_enabled(train):
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = (torch.sigmoid(logits) > 0.5)
            tp, fp, fn, _ = confusion_counts(preds, masks.bool())
            tp_sum += tp.sum().item()
            fp_sum += fp.sum().item()
            fn_sum += fn.sum().item()
            n_samples += images.size(0)

    mean_loss = total_loss / max(n_batches, 1)
    mean_iou = iou_score(torch.tensor(tp_sum), torch.tensor(fp_sum), torch.tensor(fn_sum)).item()
    mean_dice = dice_score(torch.tensor(tp_sum), torch.tensor(fp_sum), torch.tensor(fn_sum)).item()
    return mean_loss, mean_iou, mean_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=5,
                         help="epochs to train with the pretrained encoder frozen before unfreezing (transfer learning warmup)")
    parser.add_argument("--patience", type=int, default=15, help="early stopping patience on val Dice")
    parser.add_argument("--run_name", type=str, default="unet_resnet34")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ids = load_ids(os.path.join(SPLITS_DIR, "train.txt"))
    val_ids = load_ids(os.path.join(SPLITS_DIR, "val.txt"))
    print(f"train: {len(train_ids)}, val: {len(val_ids)}")

    train_ds = PlateSegmentationDataset(train_ids, args.img_size, train=True)
    val_ds = PlateSegmentationDataset(val_ids, args.img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.encoder).to(device)
    criterion = DiceBCELoss()

    def set_encoder_trainable(trainable):
        for p in model.encoder.parameters():
            p.requires_grad = trainable

    set_encoder_trainable(False)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    best_dice = -1.0
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(CKPT_DIR, f"{args.run_name}_best.pt")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(vars(args))
        mlflow.log_param("device", str(device))
        mlflow.log_param("train_size", len(train_ids))
        mlflow.log_param("val_size", len(val_ids))

        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_iou", "train_dice", "val_loss", "val_iou", "val_dice", "lr", "encoder_frozen", "seconds"])

            for epoch in range(1, args.epochs + 1):
                t0 = time.time()

                if epoch == args.freeze_encoder_epochs + 1:
                    print("unfreezing encoder (fine-tuning full network)")
                    set_encoder_trainable(True)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr / 10)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

                train_loss, train_iou, train_dice = run_epoch(model, train_loader, device, criterion, optimizer)
                val_loss, val_iou, val_dice = run_epoch(model, val_loader, device, criterion, optimizer=None)
                scheduler.step(val_dice)

                frozen = epoch <= args.freeze_encoder_epochs
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]["lr"]
                writer.writerow([epoch, train_loss, train_iou, train_dice, val_loss, val_iou, val_dice, lr_now, frozen, round(elapsed, 1)])
                f.flush()

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_iou": train_iou,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                    "lr": lr_now,
                }, step=epoch)

                print(f"epoch {epoch:03d} | train loss {train_loss:.4f} iou {train_iou:.4f} dice {train_dice:.4f} "
                      f"| val loss {val_loss:.4f} iou {val_iou:.4f} dice {val_dice:.4f} | {elapsed:.1f}s")

                if val_dice > best_dice:
                    best_dice = val_dice
                    epochs_without_improvement = 0
                    torch.save({
                        "model_state": model.state_dict(),
                        "encoder": args.encoder,
                        "img_size": args.img_size,
                        "val_dice": val_dice,
                        "val_iou": val_iou,
                        "epoch": epoch,
                    }, best_ckpt_path)
                    print(f"  -> new best val dice {val_dice:.4f}, saved to {best_ckpt_path}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= args.patience:
                        print(f"early stopping at epoch {epoch} (no val Dice improvement for {args.patience} epochs)")
                        break

        mlflow.log_metric("best_val_dice", best_dice)
        mlflow.log_artifact(best_ckpt_path)
        mlflow.log_artifact(LOG_PATH)

    print(f"training done. best val dice: {best_dice:.4f}. log: {LOG_PATH}")


if __name__ == "__main__":
    main()
