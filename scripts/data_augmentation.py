import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import albumentations as A
from PIL import Image
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Online augmentation
def get_train_transform():
    """AutoAugment + RandomErasing + Normalize"""
    albu_transform = A.Compose([
        A.PadIfNeeded(min_height=36, min_width=36, p=1.0),
        A.RandomCrop(height=32, width=32, p=1.0),
        A.HorizontalFlip(p=0.5),
    ])
    torch_transform = transforms.Compose([
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.25, scale=(0.0625, 0.1), ratio=(0.99, 1.0))
    ])

    def transform_fn(img):
        img = np.array(img)
        img = albu_transform(image=img)["image"]
        img = Image.fromarray(img)
        return torch_transform(img)

    return transform_fn

# Validation transform
def get_val_transform():
    """Validation: only Normalize + ToTensor"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def augment_dataset(*args, **kwargs):
    """
    Dummy function for backward compatibility.
    Keeps main.py import functional.
    Does nothing (all augmentation handled online).
    """
    print("[INFO] Skipping offline augmentation — using online augmentation during training.")

