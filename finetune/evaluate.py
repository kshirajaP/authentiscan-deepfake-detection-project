"""
evaluate.py
Produces the numbers your paper's Section 5 (Experimental Results) needs:
accuracy, precision, recall, F1, AUC, and a confusion matrix — for the
fine-tuned XceptionNet alone, and optionally for the full ensemble
(XceptionNet + MesoNet) so you can show the ensemble actually helps.

Usage:
    python evaluate.py --data_dir ./data --weights xception_finetuned.pth
    python evaluate.py --data_dir ./data --weights xception_finetuned.pth --with_mesonet

Output:
    metrics.json          -- numbers to quote directly in the paper
    confusion_matrix.png  -- figure to drop into Section 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from models.detector import XceptionNetDeepfake, create_mesonet4  # noqa: E402

from dataset import DeepfakeFaceDataset, XCEPTION_TRANSFORM  # noqa: E402


def load_xception(weights_path, device):
    model = XceptionNetDeepfake(pretrained=True)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_xception(model, loader, device):
    all_labels, all_preds, all_fake_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            fake_probs = probs[:, 1].cpu().numpy()
            preds = probs.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_fake_probs.extend(fake_probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_fake_probs)


def compute_metrics(labels, preds, probs, name):
    metrics = {
        "model": name,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
        "n_samples": len(labels),
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    return metrics


def plot_confusion(labels, preds, name, out_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "fake"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["real", "fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return cm.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--weights", type=str, default="xception_finetuned.pth")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = DeepfakeFaceDataset(args.data_dir, split=args.split, transform=XCEPTION_TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = load_xception(args.weights, device)
    labels, preds, probs = evaluate_xception(model, test_loader, device)

    results = {}
    results["xception_finetuned"] = compute_metrics(labels, preds, probs, "XceptionNet (fine-tuned)")
    results["xception_finetuned"]["confusion_matrix"] = plot_confusion(
        labels, preds, "XceptionNet (fine-tuned)", "confusion_matrix.png"
    )

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved metrics.json — use these numbers directly in Section 5 of the paper.")
    print("Report accuracy/precision/recall/F1/AUC in a table, and include confusion_matrix.png as a figure.")


if __name__ == "__main__":
    main()
