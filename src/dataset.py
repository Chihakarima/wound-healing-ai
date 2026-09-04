import glob
import os

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGES_DIR = "data/images"
MASKS_DIR = "data/masks_binary"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_image_path(stem):
    matches = glob.glob(os.path.join(IMAGES_DIR, f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"no image found for id {stem}")
    return matches[0]


def load_ids(split_path):
    with open(split_path) as f:
        return [line.strip() for line in f if line.strip()]


def get_train_transform(img_size):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.85, 1.18), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class PlateSegmentationDataset(Dataset):
    def __init__(self, ids, img_size, train):
        self.ids = ids
        self.transform = get_train_transform(img_size) if train else get_val_transform(img_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]
        image = cv2.imread(find_image_path(stem))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASKS_DIR, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        h = min(image.shape[0], mask.shape[0])
        w = min(image.shape[1], mask.shape[1])
        image = image[:h, :w]
        mask = mask[:h, :w]

        mask = (mask > 127).astype(np.float32)

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].unsqueeze(0), stem
