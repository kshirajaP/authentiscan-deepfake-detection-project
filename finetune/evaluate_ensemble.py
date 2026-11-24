"""
evaluate_ensemble.py
Produces the ablation table for the paper: fine-tuned XceptionNet alone,
MesoNet-4 alone, and the weighted ensemble, all evaluated on the SAME images.

Design rules (so the numbers are defensible):
  * MesoNet's output polarity and the ensemble weight are chosen on the
    VALIDATION split only. The TEST split is used once, for reporting.
  * Every metric has a 95% bootstrap confidence interval.
  * The ensemble-minus-Xception accuracy difference gets its own paired
    bootstrap interval, so you can say honestly whether the ensemble helps.

Put this file in finetune/ next to dataset.py. Usage:

    python evaluate_ensemble.py --data_dir ./data --weights xception_finetuned.pth

Outputs (in the current folder):
    ablation_metrics.json   all numbers
    ablation_table.md       table to copy into the paper
    roc_curves.png          ROC curves for the three systems
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve,
)

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))


# ----------------------------------------------------------------------
# Model outputs (heavy imports kept inside functions)
# ----------------------------------------------------------------------
def get_xception_probs(data_dir, split, weights, batch_size=32):
    """Returns (labels, P(fake), samples) for the fine-tuned Xception."""
    import torch
    from torch.utils.data import DataLoader
    from models.detector import XceptionNetDeepfake
    from dataset import DeepfakeFaceDataset, XCEPTION_TRANSFORM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DeepfakeFaceDataset(data_dir, split=split, transform=XCEPTION_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = XceptionNetDeepfake(pretrained=True)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    probs = []
    with torch.no_grad():
        for images, _ in loader:
            out = model(images.to(device))
            probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy().tolist())

    labels = np.array([l for _, l in ds.samples])
    return labels, np.array(probs), ds.samples


def get_mesonet_raw(samples, weights_path, batch_size=64):
    """Raw sigmoid output of MesoNet-4 for the given (path, label) samples.
    Preprocessing matches detector.py: RGB, 256x256, scaled to [0, 1]."""
    from PIL import Image
    import tensorflow as tf
    try:  # MesoNet is tiny: keep it on CPU so TensorFlow doesn't take the GPU from PyTorch
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    from models.detector import create_mesonet4

    model = create_mesonet4()
    model.load_weights(str(weights_path))
    out = []
    for i in range(0, len(samples), batch_size):
        batch = [
            np.asarray(Image.open(p).convert("RGB").resize((256, 256))) / 255.0
            for p, _ in samples[i:i + batch_size]
        ]
        out.extend(model.predict(np.stack(batch), verbose=0).ravel().tolist())
    return np.array(out)


# ----------------------------------------------------------------------
# Numeric helpers (no torch / tensorflow needed)
# ----------------------------------------------------------------------
def choose_polarity(val_labels, val_raw):
    """MesoNet's published weights may output P(real) rather than P(fake).
    Decide using the VALIDATION set: if AUC < 0.5, flip the output."""
    auc = roc_auc_score(val_labels, val_raw)
    flipped = auc < 0.5
    return flipped, float(auc)


def choose_weight(val_labels, val_xcep, val_meso, grid=21):
    """Weight on Xception (MesoNet gets 1 - w), picked by validation AUC."""
    best_w, best_auc = None, -1.0
    for w in np.linspace(0.0, 1.0, grid):
        auc = roc_auc_score(val_labels, w * val_xcep + (1 - w) * val_meso)
        if auc > best_auc + 1e-12:
            best_w, best_auc = float(w), float(auc)
    return best_w, best_auc


def point_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
    }


def bootstrap(labels, systems, n_boot=1000, seed=0):
    """Paired bootstrap over test images.
    systems: dict name -> P(fake). Returns (cis, acc_diffs) where cis[name][metric]
    is (low, high) and acc_diffs is a dict of paired accuracy differences."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    store = {name: {m: [] for m in ("accuracy", "precision", "recall", "f1", "auc")}
             for name in systems}
    diffs = {"ensemble_minus_xception": [], "ensemble_minus_mesonet": []}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = labels[idx]
        if len(np.unique(yb)) < 2:
            continue
        accs = {}
        for name, p in systems.items():
            m = point_metrics(yb, p[idx])
            accs[name] = m["accuracy"]
            for k, v in m.items():
                store[name][k].append(v)
        if "ensemble" in accs and "xception" in accs:
            diffs["ensemble_minus_xception"].append(accs["ensemble"] - accs["xception"])
        if "ensemble" in accs and "mesonet" in accs:
            diffs["ensemble_minus_mesonet"].append(accs["ensemble"] - accs["mesonet"])

    cis = {
        name: {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
               for k, v in mets.items()}
        for name, mets in store.items()
    }
    diff_ci = {
        k: {"mean": float(np.mean(v)),
            "ci95": (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))}
        for k, v in diffs.items() if v
    }
    return cis, diff_ci


def markdown_table(results, cis):
    lines = [
        "| Model | Accuracy (95% CI) | Precision | Recall | F1 | AUC |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lo, hi = cis[name]["accuracy"]
        lines.append(
            f"| {name} | {m['accuracy']*100:.1f}% ({lo*100:.1f}-{hi*100:.1f}) | "
            f"{m['precision']*100:.1f}% | {m['recall']*100:.1f}% | "
            f"{m['f1']:.3f} | {m['auc']:.3f} |"
        )
    return "\n".join(lines)


def plot_roc(labels, systems, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in systems.items():
        fpr, tpr, _ = roc_curve(labels, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(labels, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--weights", default="xception_finetuned.pth")
    ap.add_argument("--mesonet_weights",
                    default=str(BACKEND / "models" / "pretrained" / "mesonet4.h5"))
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    # ---- validation: choose polarity and ensemble weight ----
    print("Scoring validation split...")
    v_labels, v_xcep, v_samples = get_xception_probs(args.data_dir, args.val_split, args.weights)
    v_meso_raw = get_mesonet_raw(v_samples, args.mesonet_weights)

    flipped, raw_auc = choose_polarity(v_labels, v_meso_raw)
    v_meso = 1 - v_meso_raw if flipped else v_meso_raw
    print(f"MesoNet raw validation AUC = {raw_auc:.3f} -> "
          f"{'output is P(real); flipping' if flipped else 'output is P(fake); keeping'}")

    w, w_val_auc = choose_weight(v_labels, v_xcep, v_meso)
    print(f"Chosen Xception weight w = {w:.2f} (validation AUC {w_val_auc:.3f})")

    # ---- test: report once ----
    print("Scoring test split...")
    t_labels, t_xcep, t_samples = get_xception_probs(args.data_dir, args.test_split, args.weights)
    t_meso_raw = get_mesonet_raw(t_samples, args.mesonet_weights)
    t_meso = 1 - t_meso_raw if flipped else t_meso_raw

    systems = {
        "xception": t_xcep,
        "mesonet": t_meso,
        "ensemble": w * t_xcep + (1 - w) * t_meso,
        "ensemble_fixed_0.6_0.4": 0.6 * t_xcep + 0.4 * t_meso,
    }
    display = {
        "xception": "XceptionNet (fine-tuned)",
        "mesonet": "MesoNet-4 (pretrained)",
        "ensemble": f"Ensemble (w_X={w:.2f}, chosen on val)",
        "ensemble_fixed_0.6_0.4": "Ensemble (fixed 0.6/0.4)",
    }

    results = {k: point_metrics(t_labels, p) for k, p in systems.items()}
    cis, diff_ci = bootstrap(t_labels, systems, n_boot=args.n_boot)

    pretty = {display[k]: v for k, v in results.items()}
    pretty_ci = {display[k]: v for k, v in cis.items()}
    table = markdown_table(pretty, pretty_ci)
    print("\n" + table)
    print("\nPaired accuracy differences (bootstrap):")
    for k, v in diff_ci.items():
        print(f"  {k}: mean {v['mean']*100:+.2f} pts, "
              f"95% CI [{v['ci95'][0]*100:+.2f}, {v['ci95'][1]*100:+.2f}]")
    print("If a difference's interval contains 0, do NOT claim the ensemble improves accuracy.")

    with open("ablation_metrics.json", "w") as f:
        json.dump({
            "n_test": int(len(t_labels)),
            "mesonet_output_flipped": bool(flipped),
            "mesonet_raw_val_auc": raw_auc,
            "ensemble_weight_xception": w,
            "metrics": results,
            "bootstrap_ci95": cis,
            "paired_accuracy_difference": diff_ci,
        }, f, indent=2)
    with open("ablation_table.md", "w") as f:
        f.write(table + "\n")
    plot_roc(t_labels, {display[k]: p for k, p in systems.items() if k != "ensemble_fixed_0.6_0.4"},
             "roc_curves.png")
    print("\nSaved ablation_metrics.json, ablation_table.md, roc_curves.png")


if __name__ == "__main__":
    main()