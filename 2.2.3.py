import argparse
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from datasets import load_dataset

# -----------------------
# CLI
# -----------------------
def get_args():
    p = argparse.ArgumentParser(description="PyTorch MNIST (HF datasets)")
    p.add_argument("--dataset", type=str, default="ylecun/mnist",
                   help="Hugging Face dataset repo id")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()

# -----------------------
# Repro / Device
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    device = torch.device("cuda:0")
    return device
# -----------------------
# Data
# -----------------------
class HF_MNIST(Dataset):
    """
    Wrap a Hugging Face split (e.g., dataset['train']) as a PyTorch Dataset.
    Expects items with keys: 'image' (PIL.Image) and 'label' (int).
    """
    def __init__(self, hf_split, transform=None):
        self.split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        sample = self.split[idx]
        img = sample["image"]        # PIL.Image
        label = sample["label"]      # int
        if self.transform:
            img = self.transform(img)
        return img, label

def make_dataloaders(repo_id: str, bsz: int, test_bsz: int, num_workers: int):
    # Standard MNIST normalization
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])

    # Load splits from Hugging Face (auto-downloads & caches on first run)
    dataset = load_dataset(repo_id)

    train_ds = HF_MNIST(dataset["train"], transform=transform)
    test_ds  = HF_MNIST(dataset["test"],  transform=transform)

    # Windows note: persistent_workers=True requires num_workers>0 and Python >=3.8
    train_loader = DataLoader(
        train_ds, batch_size=bsz, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=test_bsz, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return train_loader, test_loader

# -----------------------
# Model
# -----------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.pool  = nn.MaxPool2d(2, 2)                          # halves H/W
        self.drop  = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # -> 7x7
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100.0
    dt = time.time() - t0
    return epoch_loss, epoch_acc, dt

@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss_sum += criterion(outputs, labels).item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    test_loss = loss_sum / total
    test_acc = correct / total * 100.0
    return test_loss, test_acc

# -----------------------
# Main
# -----------------------
def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = make_dataloaders(
        repo_id=args.dataset,
        bsz=args.batch_size,
        test_bsz=args.test_batch_size,
        num_workers=args.num_workers
    )

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    if scaler is not None:
        print("AMP enabled (mixed precision).")

    best_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "mnist_cnn_best.pt"

    best_path = Path(args.save_dir) / "mnist_cnn_best.pt"
    if args.eval_only:
        # charge et évalue sans ré-entraîner
        model.load_state_dict(torch.load(best_path, map_location=device))
        _, test_loader = make_dataloaders(args.dataset, args.batch_size, args.test_batch_size, args.num_workers)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"[EVAL ONLY] {best_path} | test_loss={test_loss:.4f} test_acc={test_acc:.2f}%")

        tx = T.Compose([T.Grayscale(), T.Resize((28, 28)), T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        img = Image.open("data/test5.png")  # image 28x28 fond sombre, chiffre clair (comme MNIST)
        x = tx(img).unsqueeze(0)  # shape: [1,1,28,28]

        model.eval()
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(x.to(next(model.parameters()).device))
            pred = logits.argmax(1).item()
        end = time.perf_counter()

        print(f"Prediction: {pred}")
        print(f"Inference time: {(end - start) * 1000:.3f} ms")
        return

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, dt = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.2f}%  "
            f"test_loss: {test_loss:.4f}  test_acc: {test_acc:.2f}%  "
            f"time: {dt:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ Saved new best: {best_acc:.2f}% -> {best_path}")

    print("Done.")

if __name__ == "__main__":
    main()
