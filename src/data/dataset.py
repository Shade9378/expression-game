from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.fer2013 import Record, read_image
from data.preprocessing import preprocess_rgb


def augmentation(seed: int) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ],
        seed=seed,
    )


class FERDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        size: int,
        train: bool = False,
        seed: int = 42,
        cache: bool = True,
    ) -> None:
        self.records = records
        self.size = size
        self.augment = augmentation(seed) if train else None
        self.images = [np.asarray(read_image(r)) for r in records] if cache else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = (
            self.images[index].copy()
            if self.images is not None
            else np.asarray(read_image(self.records[index]))
        )
        if self.augment is not None:
            image = self.augment(image=image)["image"]
        return torch.from_numpy(preprocess_rgb(image, self.size)), self.records[index].label
