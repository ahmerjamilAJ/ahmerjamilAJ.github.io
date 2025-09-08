# train_and_export_mnist_onnx.py
import os, math, argparse, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import numpy as np
import onnx
import onnxruntime as ort

# ---- Config (MNIST normalization; matches common training) ----
NORM_MEAN = 0.1307
NORM_STD  = 0.3081

class Net(nn.Module):
    # Simple, fast CNN good enough for MNIST
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)   # 28->26
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 26->24
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*12*12, 128)   # after 2x2 pool: 24->12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)     # 24->12
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)            # logits
        return x

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_loaders(batch_size=128, val_split=5000, root="./data"):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,))
    ])
    train_full = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test_ds    = datasets.MNIST(root, train=False, download=True, transform=tfm)

    # small validation for sanity; rest for training
    train_len = len(train_full) - val_split
    train_ds, val_ds = random_split(train_full, [train_len, val_split], generator=torch.Generator().manual_seed(42))

    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    test  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    return train, val, test

def train_one_epoch(model, loader, opt, device):
    model.train(); loss_sum = 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward(); opt.step()
        loss_sum += float(loss) * x.size(0)
    return loss_sum / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct=0; total=0; loss_sum=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction='sum'))
        pred = logits.argmax(1)
        correct += int((pred==y).sum())
        total += y.numel()
    return loss_sum/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)       # 3 is already ~97–99% on MNIST
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--onnx",   type=str, default="assets/models/mnist_cnn.onnx")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    train, val, test = get_loaders(batch_size=args.batch)

    model = Net().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

    best_val = 0.0
    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train, opt, device)
        val_loss, val_acc = evaluate(model, val, device)
        print(f"[{ep}/{args.epochs}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}

    # Load best weights (by val acc) before export
    if 'best_state' in locals():
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test, device)
    print(f"TEST: loss={test_loss:.4f}  acc={test_acc*100:.2f}%")

    # ---- Export to ONNX ----
    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 1, 28, 28)   # NCHW
    input_names = ["input"]
    output_names = ["logits"]

    torch.onnx.export(
        model_cpu, dummy, str(onnx_path),
        input_names=input_names, output_names=output_names,
        opset_version=13, dynamic_axes=None
    )
    print(f"Saved ONNX → {onnx_path}")

    # ---- Quick sanity check with onnxruntime ----
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = dummy.numpy()
    # apply same normalization the model expects
    x = (x - NORM_MEAN) / NORM_STD
    logits_onnx = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x.astype(np.float32)})[0]
    # compare with PyTorch
    with torch.no_grad():
        logits_torch = model_cpu((dummy - NORM_MEAN)/NORM_STD).numpy()
    diff = np.max(np.abs(logits_onnx - logits_torch))
    print(f"Verification: max|ONNX - Torch| = {diff:.6f} (should be < 1e-3 to 1e-2)")

if __name__ == "__main__":
    main()
