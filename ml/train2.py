#!/usr/bin/env python3
"""
GAF + RP + MTF + Raw Price Regression Trainer

This script preprocesses tick data CSVs into multi‐channel “images”:
1) Gramian Angular Field (summation)  
2) Gramian Angular Field (difference)  
3) Recurrence Plot  
4) Markov Transition Field  
5) Raw normalized price image  

It then labels them for regression based on future price movements
and trains a five‐channel MobileNetV3‐based regression model.
"""
import argparse
import logging
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm.auto import tqdm


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(stream=sys.stdout, level=level, format=fmt)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multi‐channel GAF+RP+MTF+Raw regression on tick CSVs"
    )
    parser.add_argument("input_dir", help="Directory of tick CSVs")
    parser.add_argument("data_dir", help="Where to save/load .npz tensors")
    parser.add_argument("labels_dir", help="Where to save label CSVs")
    parser.add_argument(
        "--freq", default="1S", help="Resample frequency (e.g. '1S')"
    )
    parser.add_argument(
        "--window", default="5T", help="Window length (e.g. '5T')"
    )
    parser.add_argument(
        "--stride", default="1S", help="Stride between windows"
    )
    parser.add_argument(
        "--horizon", default="5T", help="Future horizon for labeling"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--wd", type=float, default=1e-5, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of DataLoader workers; defaults to CPU count"
    )
    parser.add_argument(
        "--model_out", default="best_5c_regression.pth",
        help="Output path for best model weights"
    )
    parser.add_argument(
        "--t0", type=int, default=5,
        help="Epochs for first restart in CosineAnnealingWarmRestarts"
    )
    return parser.parse_args()


def normalize(series: np.ndarray) -> np.ndarray:
    """Scale array to [-1, 1]; returns zeros if constant."""
    mn, mx = series.min(), series.max()
    if mn == mx:
        return np.zeros_like(series, dtype=np.float32)
    scaled = 2 * (series - mn) / (mx - mn) - 1
    return scaled.astype(np.float32)


def load_csv(fp: Path) -> pd.DataFrame:
    """Load and standardize price CSVs to DataFrame with 'close' and datetime index."""
    df = pd.read_csv(fp)
    # Rename common columns if necessary
    if "price" in df.columns and "time" in df.columns:
        df = df.rename(columns={"price": "close", "time": "timestamp"})
    # Ensure timestamp column
    if "timestamp" not in df.columns:
        df.columns = ["timestamp"] + list(df.columns[1:])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if "close" not in df.columns:
        raise KeyError(f"Missing 'close' column in {fp}")
    return df


def preprocess(
    input_dir: Path,
    data_dir: Path,
    freq: str,
    window: str,
    stride: str,
    logger: logging.Logger,
) -> None:
    """Convert tick CSVs to multi‐channel .npz files for each sliding window."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Transformers for each channel
    gaf_summ = GramianAngularField(method="summation")
    gaf_diff = GramianAngularField(method="difference")
    rp_tf = RecurrencePlot()
    mtf_tf = MarkovTransitionField()

    w_bars = int(pd.Timedelta(window) / pd.Timedelta(freq))
    s_bars = max(int(pd.Timedelta(stride) / pd.Timedelta(freq)), 1)

    logger.info("Starting preprocessing: %d bars/window, stride %d bars", w_bars, s_bars)

    for fp in sorted(input_dir.glob("*.csv")):
        try:
            df = load_csv(fp)
            series = (
                df["close"]
                .resample(freq)
                .last()
                .ffill()
                .to_numpy(dtype=np.float32)
            )
        except Exception as e:
            logger.warning("Skipping %s: %s", fp.name, e)
            continue

        n = len(series)
        for i in range(0, n - w_bars + 1, s_bars):
            out_fp = data_dir / f"{fp.stem}_win{i:05d}.npz"
            if out_fp.exists():
                continue

            window_arr = series[i : i + w_bars]
            normed = normalize(window_arr)

            # 1) Summation GAF
            sum_gaf = gaf_summ.fit_transform(normed.reshape(1, -1))[0]
            # 2) Difference GAF
            diff_gaf = gaf_diff.fit_transform(normed.reshape(1, -1))[0]
            # 3) Recurrence Plot
            rp = rp_tf.fit_transform(normed.reshape(1, -1))[0]
            # 4) Markov Transition Field
            mtf = mtf_tf.fit_transform(normed.reshape(1, -1))[0]
            # 5) Raw normalized price image
            raw_price_image = np.tile(normed, (w_bars, 1))

            # Stack into 5-channel array: [GAF_sum, GAF_diff, RP, MTF, RawPrice]
            img_5c = np.stack([sum_gaf, diff_gaf, rp, mtf, raw_price_image], axis=0)
            np.savez_compressed(out_fp, data=img_5c)

        logger.debug("Processed %s", fp.name)


def label(
    input_dir: Path,
    labels_dir: Path,
    freq: str,
    window: str,
    stride: str,
    horizon: str,
    logger: logging.Logger,
) -> None:
    """Generate CSV label files mapping windows to future returns."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    w_bars = int(pd.Timedelta(window) / pd.Timedelta(freq))
    s_bars = max(int(pd.Timedelta(stride) / pd.Timedelta(freq)), 1)
    h_bars = int(pd.Timedelta(horizon) / pd.Timedelta(freq))

    logger.info("Starting labeling: window %d, horizon %d bars", w_bars, h_bars)

    for fp in sorted(input_dir.glob("*.csv")):
        out_fp = labels_dir / f"{fp.stem}_labels.csv"
        if out_fp.exists():
            continue
        try:
            df = load_csv(fp)
        except Exception as e:
            logger.warning("Skipping %s: %s", fp.name, e)
            continue

        series = (
            df["close"].resample(freq).last().ffill().to_numpy(dtype=np.float32)
        )
        rows = []
        for i in range(0, len(series) - w_bars + 1, s_bars):
            end = i + w_bars - 1
            if end + h_bars >= len(series):
                break
            p0, pf = series[end], series[end + h_bars]
            ret = (pf - p0) / p0 if p0 else 0.0
            rows.append({
                "filename": f"{fp.stem}_win{i:05d}.npz",
                "label": ret,
            })
        pd.DataFrame(rows).to_csv(out_fp, index=False)
        logger.debug("Labeled %s: %d windows", fp.name, len(rows))


class KlineDataset(Dataset):
    """PyTorch Dataset for multi‐channel regression windows."""

    def __init__(self, labels_df: pd.DataFrame, data_dir: Path):
        self.df = labels_df.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        arr = np.load(self.data_dir / row.filename)["data"]
        x = torch.from_numpy(arr).float()
        y = torch.tensor(row.label, dtype=torch.float32)
        return x, y


def build_model(device: torch.device) -> nn.Module:
    """Instantiate and return the regression model (5‐channel input)."""
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    # Adapt first convolution for 5-channel input
    orig_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=5,
        out_channels=orig_conv.out_channels,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        bias=False,
    )
    # Adapt classifier for single-output regression
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 1)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    model.to(device).to(memory_format=torch.channels_last)
    return model


def train(
    labels_dir: Path,
    data_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Train the regression model and save the best checkpoint."""
    # Load and concatenate all label files
    label_files = sorted(labels_dir.glob("*_labels.csv"))
    if not label_files:
        logger.error("No label files found in %s", labels_dir)
        return
    df = pd.concat([pd.read_csv(fp) for fp in label_files], ignore_index=True)
    df = df.sort_values("filename").reset_index(drop=True)

    # Split data
    n_total  = len(df)
    n_train  = int(n_total * 0.8)
    n_val    = int(n_total * 0.1)
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train : n_train + n_val]

    # DataLoaders
    num_workers = args.workers or os.cpu_count()
    train_loader = DataLoader(
        KlineDataset(train_df, data_dir),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        KlineDataset(val_df, data_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    model     = build_model(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    scheduler_cos     = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.t0, T_mult=1, eta_min=args.lr * 1e-2
    )
    scheduler_plateau = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss  = 0.0
        train_count = 0
        desc = f"Epoch {epoch}/{args.epochs} [Train]"
        for x_batch, y_batch in tqdm(train_loader, desc=desc):
            x_batch = x_batch.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(x_batch).squeeze(1)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            scheduler_cos.step(epoch + train_count / len(train_loader))

            bs = y_batch.size(0)
            train_loss  += loss.item() * bs
            train_count += bs

        # Validation
        model.eval()
        val_loss  = 0.0
        val_count = 0
        for x_batch, y_batch in tqdm(val_loader, desc="Validation"):
            x_batch = x_batch.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y_batch = y_batch.to(device, non_blocking=True)
            with torch.no_grad():
                preds = model(x_batch).squeeze(1)
                loss  = criterion(preds, y_batch)
            val_loss  += loss.item() * y_batch.size(0)
            val_count += y_batch.size(0)

        val_mse = val_loss / val_count
        logger.info("Epoch %d: Val MSE=%.6f", epoch, val_mse)
        scheduler_plateau.step(val_mse)

        # Checkpoint
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), args.model_out)
            logger.info("New best model saved: %s (MSE=%.6f)", args.model_out, best_val_mse)

    logger.info("Training complete. Best Val MSE=%.6f", best_val_mse)


def main() -> None:
    setup_logging()
    args       = parse_args()
    input_dir  = Path(args.input_dir)
    data_dir   = Path(args.data_dir)
    labels_dir = Path(args.labels_dir)
    logger     = logging.getLogger(__name__)

    try:
        preprocess(input_dir, data_dir, args.freq, args.window, args.stride, logger)
        label(input_dir, labels_dir, args.freq, args.window, args.stride, args.horizon, logger)
        train(labels_dir, data_dir, args, logger)
    except Exception as e:
        logger.exception("Execution error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()