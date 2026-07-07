"""
Training script for ICU mortality prediction.

Usage
-----
# Synthetic data, LSTM:
python train.py --model lstm --data synthetic

# Synthetic data, TFT:
python train.py --model tft --data synthetic

# Real MIMIC-III:
python train.py --model tft --data mimic --mimic_path /path/to/mimic

Options
-------
--model         : lstm | tft
--data          : synthetic | mimic
--mimic_path    : path to MIMIC-III directory (required if --data mimic)
--epochs        : training epochs (default 50)
--batch_size    : (default 64)
--lr            : learning rate (default 1e-3)
--d_model       : TFT embedding dimension (default 64)
--hidden_dim    : LSTM hidden units (default 128)
--dropout       : dropout rate (default 0.3)
--pos_weight    : BCE positive class weight to handle imbalance (default auto)
--seed          : random seed (default 42)
--save_dir      : checkpoint output directory (default checkpoints/)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# local imports
sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import ICUDataset
from models.lstm import LSTMMortality
from models.tft  import TFTMortality
from evaluate    import evaluate_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    classes, counts = np.unique(y, return_counts=True)
    class_weights = 1.0 / counts
    weights = class_weights[y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )


def load_data(args):
    if args.data == "synthetic":
        from data.synthetic import generate_synthetic, normalise
        X, y = generate_synthetic(n_samples=3000, seed=args.seed)
        print(f"Synthetic dataset: {X.shape}, mortality={y.mean():.2%}")
    elif args.data == "mimic":
        from data.mimic_loader import load_mimic, normalise
        X, y = load_mimic(args.mimic_path)
    else:
        raise ValueError(f"Unknown --data: {args.data}")

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=args.seed
    )

    X_train, X_val, X_test, mean, std = normalise(X_train, X_val, X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, mean, std


def build_model(args, device):
    n_features = 5
    if args.model == "lstm":
        model = LSTMMortality(
            input_dim=n_features,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            dropout=args.dropout,
        )
    elif args.model == "tft":
        model = TFTMortality(
            input_dim=n_features,
            d_model=args.d_model,
            num_heads=4,
            num_lstm_layers=2,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown --model: {args.model}")
    return model.to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, X_val, X_test, y_train, y_val, y_test, mean, std = load_data(args)

    train_ds = ICUDataset(X_train, y_train)
    val_ds   = ICUDataset(X_val,   y_val)
    test_ds  = ICUDataset(X_test,  y_test)

    sampler = make_weighted_sampler(y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,  num_workers=0)

    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()} — {n_params:,} trainable parameters")

    # Auto pos_weight from training label ratio
    pos_weight_val = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    if args.pos_weight is not None:
        pos_weight_val = args.pos_weight
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"BCE pos_weight: {pos_weight_val:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_auroc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_ds)

        # --- Validate ---
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        scheduler.step(val_metrics["auroc"])

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} | "
            f"val_auprc={val_metrics['auprc']:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            ckpt_path = save_dir / f"{args.model}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "norm_mean": mean.tolist(),
                    "norm_std":  std.tolist(),
                },
                ckpt_path,
            )
            print(f"  * Saved best checkpoint (AUROC {best_auroc:.4f})")

    # --- Test ---
    print("\n--- Test set evaluation ---")
    ckpt = torch.load(save_dir / f"{args.model}_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_model(model, test_loader, device, criterion, detailed=True)
    print(json.dumps(test_metrics, indent=2))

    results_path = save_dir / f"{args.model}_test_results.json"
    with open(results_path, "w") as f:
        json.dump({"test": test_metrics, "history": history, "args": vars(args)}, f, indent=2)
    print(f"\nResults saved to {results_path}")
    return test_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ICU mortality prediction trainer")
    p.add_argument("--model",      default="tft", choices=["lstm", "tft"])
    p.add_argument("--data",       default="synthetic", choices=["synthetic", "mimic"])
    p.add_argument("--mimic_path", default=os.environ.get("MIMIC_PATH", ""))
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--d_model",    type=int,   default=64)
    p.add_argument("--hidden_dim", type=int,   default=128)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--pos_weight", type=float, default=None)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save_dir",   default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
