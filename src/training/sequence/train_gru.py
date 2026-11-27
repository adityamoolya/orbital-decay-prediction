"""
train_gru.py

Trains a small 1-2 layer GRU on raw 20-epoch sliding windows to predict
log1p(days_to_decay). Uses an 80/20 satellite-level validation split carved
strictly from the 156 training satellites (never touching test satellites).

Saves:
  - src/training/sequence/gru_model.pt   (best model state_dict)
  - src/training/sequence/gru_meta.npz   (train-only feature scaler + model config)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEQ_SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SEQ_SRC_DIR.parent.parent.parent
SEQ_DATA_DIR = BASE_DIR / "data" / "sequence"
MODEL_PATH = SEQ_SRC_DIR / "gru_model.pt"
META_PATH = SEQ_SRC_DIR / "gru_meta.npz"


class SatelliteGRU(nn.Module):
    """Small GRU regressor mapping (batch, seq_len, n_features) -> (batch,)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU on raw satellite TLE windows.")
    parser.add_argument("--hidden-size", type=int, default=32, help="GRU hidden size (e.g. 32 or 64)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GRU layers (1 or 2)")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction of train satellites for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_path = SEQ_DATA_DIR / "train_windows.npy"
    y_path = SEQ_DATA_DIR / "train_targets.npy"
    ids_path = SEQ_DATA_DIR / "train_sat_ids.npy"
    if not (X_path.exists() and y_path.exists() and ids_path.exists()):
        raise FileNotFoundError(
            f"Windowed training files not found in {SEQ_DATA_DIR}. Run build_windows.py first."
        )

    X_all = np.load(X_path)
    y_all = np.load(y_path)
    sat_ids_all = np.load(ids_path)

    # Carve validation split strictly by satellite ID from train satellites only
    unique_sats = np.unique(sat_ids_all)
    rng = np.random.default_rng(args.seed)
    shuffled_sats = rng.permutation(unique_sats)
    n_val_sats = max(1, int(round(len(shuffled_sats) * args.val_frac)))
    val_sats = set(shuffled_sats[:n_val_sats].tolist())
    tr_sats = set(shuffled_sats[n_val_sats:].tolist())

    tr_mask = np.isin(sat_ids_all, list(tr_sats))
    val_mask = np.isin(sat_ids_all, list(val_sats))

    X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]

    print(
        f"Train-internal split: {len(tr_sats)} train satellites ({len(X_tr):,} windows) | "
        f"{len(val_sats)} val satellites ({len(X_val):,} windows)"
    )

    # Standardize features using train-fold statistics only
    feat_mean = X_tr.mean(axis=(0, 1), keepdims=True)
    feat_std = X_tr.std(axis=(0, 1), keepdims=True)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)

    X_tr_scaled = ((X_tr - feat_mean) / feat_std).astype(np.float32)
    X_val_scaled = ((X_val - feat_mean) / feat_std).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_scaled), torch.from_numpy(y_tr)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val)),
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    input_size = X_all.shape[2]
    model = SatelliteGRU(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: SatelliteGRU(input_size={input_size}, hidden_size={args.hidden_size}, "
        f"num_layers={args.num_layers}, params={n_params:,})"
    )
    print("-" * 72)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state_dict = None
    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        tr_loss_sum = 0.0
        tr_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item() * xb.size(0)
            tr_count += xb.size(0)

        train_mse = tr_loss_sum / tr_count

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_count += xb.size(0)

        val_mse = val_loss_sum / val_count
        ep_time = time.time() - t_ep

        improved = val_mse < best_val_loss
        if improved:
            best_val_loss = val_mse
            best_epoch = epoch
            epochs_no_improve = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            epochs_no_improve += 1
            marker = ""

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_mse={train_mse:.5f} | val_mse={val_mse:.5f} | "
            f"time={ep_time:.2f}s{marker}"
        )

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch} (best epoch: {best_epoch}, val_mse={best_val_loss:.5f}).")
            break

    total_time = time.time() - t_train_start
    print("-" * 72)
    print(f"Training finished in {total_time:.2f}s | Best val_mse={best_val_loss:.5f} at epoch {best_epoch}")

    if best_state_dict is None:
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state_dict, MODEL_PATH)
    np.savez(
        META_PATH,
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        input_size=np.array(input_size, dtype=np.int64),
        hidden_size=np.array(args.hidden_size, dtype=np.int64),
        num_layers=np.array(args.num_layers, dtype=np.int64),
    )
    print(f"Saved model state_dict to {MODEL_PATH}")
    print(f"Saved scaler/config metadata to {META_PATH}")


if __name__ == "__main__":
    main()
