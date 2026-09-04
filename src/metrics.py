import torch

EPS = 1e-7


def confusion_counts(pred_binary, target_binary):
    """pred_binary, target_binary: bool tensors, any shape. Returns per-sample
    tp, fp, fn, tn summed over all dims except dim 0 (batch)."""
    pred = pred_binary.flatten(1)
    target = target_binary.flatten(1)
    tp = (pred & target).sum(1).float()
    fp = (pred & ~target).sum(1).float()
    fn = (~pred & target).sum(1).float()
    tn = (~pred & ~target).sum(1).float()
    return tp, fp, fn, tn


def iou_score(tp, fp, fn):
    return (tp + EPS) / (tp + fp + fn + EPS)


def dice_score(tp, fp, fn):
    return (2 * tp + EPS) / (2 * tp + fp + fn + EPS)


def precision_score(tp, fp):
    return (tp + EPS) / (tp + fp + EPS)


def recall_score(tp, fn):
    return (tp + EPS) / (tp + fn + EPS)


class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        pred = probs.flatten(1)
        tgt = target.flatten(1)
        intersection = (pred * tgt).sum(1)
        dice = (2 * intersection + EPS) / (pred.sum(1) + tgt.sum(1) + EPS)
        return bce + (1 - dice.mean())
