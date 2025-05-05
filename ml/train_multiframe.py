#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a CNN on 3-channel GAF images (-1,0,1 mapped to 0,1,2)')
    parser.add_argument('labels_dir', type=str,
                        help='Directory containing *_labels.csv files')
    parser.add_argument('img_dir', type=str,
                        help='Directory containing 3-channel GAF .png images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (adjust for memory)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of DataLoader workers')
    parser.add_argument('--model_out', type=str, default='best_multiframe.pth',
                        help='Where to save the best model weights')
    parser.add_argument(
        '--max_train_batches', type=int, default=None,
        help='Limit number of batches per training epoch (for quick debug)'
    )
    return parser.parse_args()

class MultiFrameDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row.filename
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)
        orig_label = int(row.label)
        y = orig_label + 1
        return x, y


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, scheduler, max_batches=None):
    model.train()
    pbar = tqdm(loader, desc='Train', leave=False)
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    autocast_dtype = torch.float16 if device.type=='cuda' else torch.bfloat16 if device.type=='mps' else None
    for batch_idx, (x, y) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=(scaler is not None or device.type=='mps')):
            out = model(x)
            loss = criterion(out, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        batch = y.size(0)
        running_loss += loss.item() * batch
        preds = out.argmax(dim=1)
        running_correct += (preds==y).sum().item()
        running_total += batch
        pbar.set_postfix({
            'loss': f'{running_loss/running_total:.4f}',
            'acc':  f'{running_correct/running_total:.3f}'
        })
    return running_loss/running_total, running_correct/running_total


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
            correct += (preds==y).sum().item()
            total += y.size(0)
            pbar.set_postfix({'acc': f'{correct/total:.3f}'})
    return correct/total


def main():
    args = parse_args()
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # load labels
    labels_dir = Path(args.labels_dir)
    csvs = sorted(labels_dir.glob('*_labels.csv'))
    if not csvs:
        raise FileNotFoundError(f'No label CSVs in {labels_dir}')
    dfs = [pd.read_csv(f) for f in csvs]
    df = pd.concat(dfs, ignore_index=True).sort_values('filename').reset_index(drop=True)
    n = len(df)
    train_end = int(n*0.8)
    val_end   = int(n*0.9)
    train_df, val_df, test_df = df[:train_end], df[train_end:val_end], df[val_end:]
    print(f'Splits → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')

    # transforms & loaders
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([transforms.ToTensor()])
    train_ds = MultiFrameDataset(train_df, args.img_dir, transform=train_tf)
    val_ds   = MultiFrameDataset(val_df,   args.img_dir, transform=val_tf)
    test_ds  = MultiFrameDataset(test_df,  args.img_dir, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=False,
                              persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=False,
                              persistent_workers=True, prefetch_factor=2)

    # model setup
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # adapt first conv for 3-channel
    # here input images already 3-channel (RGB)
    # replace classifier
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, 3)
    model = model.to(device)
    if device.type=='cuda':
        model = torch.compile(model)
    else:
        print(f'Skipping compile on {device}')
    model = model.to(memory_format=torch.channels_last)
    for param in model.features.parameters(): param.requires_grad=False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           epochs=args.epochs, steps_per_epoch=steps,
                           pct_start=0.3, div_factor=25.0, final_div_factor=1e4)
    scaler = torch.cuda.amp.GradScaler() if device.type=='cuda' else None

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader,
            criterion, optimizer, scaler, device, scheduler,
            args.max_train_batches
        )
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc>best_val:
            best_val=val_acc
            torch.save(model.state_dict(), args.model_out)
            print(f'Saved best model to {args.model_out}')

    # test
    model.load_state_dict(torch.load(args.model_out, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f'Test Acc: {test_acc:.4f}')

if __name__=='__main__':
    main()
