#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a CNN on 3-class GAF images (-1,0,1 mapped to 0,1,2)')
    parser.add_argument('labels_dir', type=str,
                        help='Directory containing *_labels.csv files')
    parser.add_argument('img_dir', type=str,
                        help='Directory containing GAF .png images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (adjust for MPS memory)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of DataLoader workers (Mac M3 Air optimized)')
    parser.add_argument('--model_out', type=str, default='best_model_mps.pth',
                        help='Where to save the best model weights')
    return parser.parse_args()

class GAFDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row.filename
        img = Image.open(img_path).convert('L')
        x = self.transform(img)
        orig_label = int(row.label)
        y = orig_label + 1
        return x, y


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    pbar = tqdm(loader, desc='Train', leave=False)
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            out = model(x)
            loss = criterion(out, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        preds = out.argmax(dim=1)
        running_correct += (preds == y).sum().item()
        running_total += batch_size
        avg_loss = running_loss / running_total
        avg_acc = running_correct / running_total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.3f}'})
    total_loss = running_loss / running_total
    total_acc = running_correct / running_total
    return total_loss, total_acc


def evaluate(model, loader, device):
    model.eval()
    pbar = tqdm(loader, desc='Eval', leave=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            pbar.set_postfix({'acc': f'{correct/total:.3f}'})
    return correct / total


def main():
    args = parse_args()
    # Device selection: prefer MPS on Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load and combine label CSVs
    labels_dir = Path(args.labels_dir)
    csv_files = sorted(labels_dir.glob('*_labels.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No label CSVs in {labels_dir}")
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('filename').reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)
    train_df  = df.iloc[:train_end]
    val_df    = df.iloc[train_end:val_end]
    test_df   = df.iloc[val_end:]
    print(f"Splits → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Transforms and DataLoaders
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([transforms.ToTensor()])

    train_ds = GAFDataset(train_df, args.img_dir, transform=train_transforms)
    val_ds   = GAFDataset(val_df,   args.img_dir, transform=val_transforms)
    test_ds  = GAFDataset(test_df,  args.img_dir, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=False)

    # Model and optimizer
    # Use MobileNet V3 Small for transfer learning
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Adapt first conv to accept single-channel input
    orig_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, orig_conv.out_channels,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        bias=False
    )
    # Replace classifier head for 3 classes
    # classifier: [Linear, Hardswish, Dropout, Linear]
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Mixed precision
    if device.type in ('cuda', 'mps'):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, scaler, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_out)
            print(f"Saved best model to {args.model_out}")

    # Final test evaluation
    model.load_state_dict(torch.load(args.model_out, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    main()