import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from scripts.data_augmentation import get_train_transform, get_val_transform
import random
import numpy as np

def load_transforms():
    """
    Load the data transformations
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_data(data_dir, batch_size):
    """
    Load CIFAR dataset with online augmentation
    (no need to generate augmented files)
    """
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"[INFO] Train set: {len(train_dataset)} samples | Online augmentation enabled")
    print(f"[INFO] Val set: {len(val_dataset)} samples | No augmentation (normalize only)")

    return train_loader, val_loader


def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float, num_epochs: int = 300, warmup_epochs: int = 20):
    """
        Define loss function, optimizer, and learning rate scheduler.
        Implements 5-epoch warmup + Cosine Annealing LR Scheduler (Strong Baseline style).
        Args:
            model: torch.nn.Module - model to train
            lr: float - initial (target) learning rate
            weight_decay: float - L2 regularization
            num_epochs: int - total epochs
            warmup_epochs: int - number of warmup epochs (default: 5)
        Returns:
            criterion, optimizer, scheduler
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    # linearly increase LR from 0 → lr over warmup_epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,     # start from 0.001 * lr
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    # Cosine annealing: smooth decay from lr → eta_min=1e-6
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(num_epochs - warmup_epochs),
        eta_min=1e-6
    )

    # Combine them sequentially
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    return criterion, optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm: float = 1.0):
    """
    Train the model for one epoch
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    # Randomly select an augmentation strategy once per epoch
    use_mixup = random.random() < 0.5
    use_cutmix = not use_mixup

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad(set_to_none=True)

        # Mixup 或 CutMix
        if use_mixup:
            lam = np.random.beta(1.0, 1.0)
            index = torch.randperm(inputs.size(0)).to(device)
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
            labels_a, labels_b = labels, labels[index]

        elif use_cutmix:
            lam = np.random.beta(1.0, 1.0)
            index = torch.randperm(inputs.size(0)).to(device)
            W, H = inputs.size(2), inputs.size(3)
            cut_rat = np.sqrt(1. - lam)
            cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
            cx, cy = np.random.randint(W), np.random.randint(H)
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            mixed_inputs = inputs.clone()
            mixed_inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            labels_a, labels_b = labels, labels[index]
        else:
            mixed_inputs, labels_a, labels_b, lam = inputs, labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)


        # Forward pass
        outputs = model(inputs)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

        # Backward pass and optimize
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    Returns:
        Average loss and accuracy for the validation set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    Args:
        state: Checkpoint state
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
    Returns:
        Checkpoint state
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint

def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """
    Save training metrics to a file
    Args:
        metrics: Metrics string to save
        filename: Path to save metrics
    """
    with open(filename, 'w') as f:
        f.write(metrics)
