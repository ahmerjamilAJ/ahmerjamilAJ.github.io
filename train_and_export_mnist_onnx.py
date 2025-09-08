#!/usr/bin/env python3
"""
Train a more robust MNIST CNN and export to ONNX for onnxruntime-web.

Input (unchanged):
  - Tensor: [N, 1, 28, 28]
  - Normalize: mean=0.1307, std=0.3081

Output (unchanged):
  - logits: [N, 10]

ONNX location (unchanged):
  assets/models/mnist_cnn.onnx
"""

import argparse, math, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

# -----------------------
# Repro
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------
# Model (same I/O shape)
# -----------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# -----------------------
# Data
# -----------------------
MEAN, STD = 0.1307, 0.3081  # must match your HTML

def build_dataloaders(batch_size: int, num_workers: int, extra_train: bool):
    # Strong-but-safe augments that keep digit semantics and 28x28 shape
    train_tfms = transforms.Compose([
        transforms.RandomApply([transforms.RandomAffine(
            degrees=15, translate=(0.12, 0.12), scale=(0.9, 1.1), shear=8
        )], p=0.85),
        transforms.RandomApply([transforms.ElasticTransform(alpha=20.0, sigma=5.0)], p=0.25),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])

    # Core MNIST
    train1 = datasets.MNIST(root="./data", train=True, download=True, transform=train_tfms)
    # Optionally double training set by loading a second copy (different augment randomness)
    train2 = datasets.MNIST(root="./data", train=True, download=True, transform=train_tfms) if extra_train else None
    train_ds = ConcatDataset([d for d in [train1, train2] if d is not None])

    val_ds = datasets.MNIST(root="./data", train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# -----------------------
# EMA (for a steadier final model)
# -----------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])

# -----------------------
# Train / Eval
# -----------------------
def evaluate(model, loader, device):
    model.eval()
    correct, n = 0, 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            n += y.size(0)
    return loss_sum / n, correct / n

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, val_loader = build_dataloaders(args.batch_size, args.workers, args.extra_train)

    model = SmallCNN().to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # OneCycleLR for fast, stable convergence
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.15, div_factor=10, final_div_factor=100
    )

    # Label smoothing helps generalization on messy drawings
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = EMA(model, decay=0.999)

    best_acc, best_state = 0.0, None
    patience, bad = 6, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            # Gradient clipping guards against occasional augmentation outliers
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            epoch_loss += loss.item() * y.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device)

        # Track best raw model (we'll also consider EMA below)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={epoch_loss/len(train_loader.dataset):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% "
              f"lr={sched.get_last_lr()[0]:.2e}")

        if bad >= patience:
            print("Early stopping: no validation improvement.")
            break

    # Choose between best raw vs EMA snapshot (use whichever validates higher)
    # Evaluate EMA
    ema.store(model)
    ema.copy_to(model)
    ema_loss, ema_acc = evaluate(model, val_loader, device)
    ema.restore(model)

    if ema_acc >= best_acc:
        print(f"Using EMA weights for export (val_acc={ema_acc*100:.2f}% ≥ best_raw={best_acc*100:.2f}%).")
        ema.copy_to(model)
    else:
        print(f"Using best raw weights for export (best_raw={best_acc*100:.2f}% > ema={ema_acc*100:.2f}%).")
        model.load_state_dict(best_state)

    # Export ONNX to the SAME path your service expects
    onnx_path = Path("assets/models/mnist_cnn.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13, do_constant_folding=True
    )
    dt = time.time() - t0
    print(f"Saved ONNX → {onnx_path.resolve()} (elapsed {dt/60:.1f} min)")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra_train", action="store_true",
                   help="Double effective train size by adding a second augmented pass")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
