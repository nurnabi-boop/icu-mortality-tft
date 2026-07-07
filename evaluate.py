"""
Evaluation utilities: AUROC, AUPRC, calibration, classification report.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    brier_score_loss,
)
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    detailed: bool = False,
) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits, _ = model(X_batch)
            if criterion is not None:
                loss = criterion(logits, y_batch.float())
                total_loss += loss.item() * len(y_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs_np  = torch.sigmoid(torch.from_numpy(logits_np)).numpy()

    metrics = {
        "auroc": float(roc_auc_score(labels_np, probs_np)),
        "auprc": float(average_precision_score(labels_np, probs_np)),
        "brier": float(brier_score_loss(labels_np, probs_np)),
    }
    if criterion is not None:
        metrics["loss"] = total_loss / len(loader.dataset)

    if detailed:
        threshold = find_best_threshold(labels_np, probs_np)
        preds = (probs_np >= threshold).astype(int)
        report = classification_report(labels_np, preds, target_names=["survived", "died"], output_dict=True)
        metrics["threshold"] = float(threshold)
        metrics["classification_report"] = report
        metrics["n_positive"] = int(labels_np.sum())
        metrics["n_total"]    = int(len(labels_np))
        print(f"\nClassification report (threshold={threshold:.3f}):")
        print(classification_report(labels_np, preds, target_names=["survived", "died"]))

    return metrics


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises F1 on the positive class."""
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def plot_results(results_path: str, save_dir: str = "results"):
    """Generate AUROC / calibration / attention plots from saved results JSON."""
    import json, os
    from pathlib import Path

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        print("No training history to plot.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs  = [h["epoch"] for h in history]
        t_loss  = [h["train_loss"] for h in history]
        v_loss  = [h.get("loss", float("nan")) for h in history]
        v_auroc = [h["auroc"] for h in history]
        v_auprc = [h["auprc"] for h in history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, t_loss, label="train")
        axes[0].plot(epochs, v_loss, label="val")
        axes[0].set_title("Loss")
        axes[0].legend()

        axes[1].plot(epochs, v_auroc, color="steelblue")
        axes[1].set_title("Val AUROC")

        axes[2].plot(epochs, v_auprc, color="darkorange")
        axes[2].set_title("Val AUPRC")

        fig.tight_layout()
        out = Path(save_dir) / "training_curves.png"
        fig.savefig(out, dpi=120)
        print(f"Saved training curves to {out}")
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed — skipping plots")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        plot_results(sys.argv[1])
