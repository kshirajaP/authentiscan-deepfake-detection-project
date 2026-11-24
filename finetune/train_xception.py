"""
train_xception.py
Fine-tunes the XceptionNet classifier head (and optionally the last
backbone block) on a real/fake face dataset.

This is the piece that was MISSING from the original repo: previously
XceptionNetDeepfake's final Linear(512,2) layer was randomly
initialized and never trained on deepfake data.

Usage:
    python train_xception.py --data_dir ./data --epochs 10 --unfreeze_last_block

Output:
    xception_finetuned.pth  (state_dict, drop into backend/models/pretrained/)
    training_log.json       (loss/accuracy per epoch, useful for the paper)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make backend/ importable so we reuse the EXACT same model class
# used at inference time (keeps architecture in sync).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from models.detector import XceptionNetDeepfake  # noqa: E402

from dataset import DeepfakeFaceDataset  # noqa: E402


def build_model(device, unfreeze_last_block=False):
    model = XceptionNetDeepfake(pretrained=True)

    # Freeze the whole backbone by default -- with a small dataset,
    # fine-tuning only the head avoids overfitting and trains fast.
    for param in model.base_model.parameters():
        param.requires_grad = False

    if unfreeze_last_block:
        # timm's xception exposes named blocks; unfreeze the last conv
        # block + exit flow so the model can adapt low-level features
        # to deepfake artifacts, not just ImageNet semantics.
        for name, param in model.base_model.named_parameters():
            if any(key in name for key in ["block12", "conv3", "conv4", "bn3", "bn4"]):
                param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze_last_block", action="store_true",
                         help="Also fine-tune the last Xception block, not just the head")
    parser.add_argument("--output", type=str, default="xception_finetuned.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected. This will be very slow — "
              "use a Colab GPU runtime or a machine with CUDA.")

    train_ds = DeepfakeFaceDataset(args.data_dir, split="train")
    val_ds = DeepfakeFaceDataset(args.data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(device, unfreeze_last_block=args.unfreeze_last_block)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_acc = 0.0
    log = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

        log.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "seconds": dt
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> new best (val_acc={val_acc:.4f}), saved to {args.output}")

    with open("training_log.json", "w") as f:
        json.dump({"best_val_acc": best_val_acc, "epochs": log}, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Weights saved to: {args.output}")
    print(f"Log saved to: training_log.json  (use this for the paper's training curve)")


if __name__ == "__main__":
    main()
