import os
from pathlib import Path
from functools import lru_cache

import dask.dataframe as dd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint

import streamlit as st

# -- Parquet Loader with Python Cache ---------------------------------------
@lru_cache(maxsize=None)
def load_parquet_numbers(file_path: str):
    ddf = dd.read_parquet(file_path, columns=None)
    df = ddf.select_dtypes(include=['number']).compute()
    return df

# -- Config Loader ---------------------------------------------------------
def load_config_from_dict(cfg_dict):
    numeric_keys = {
        'seq_len': int,
        'batch_size': int,
        'num_workers': int,
        'd_model': int,
        'n_heads': int,
        'num_layers': int,
        'diff_steps': int,
        'epochs': int,
        'ckpt_every': int,
        'step_size': int
    }
    float_keys = ['lr', 'gamma', 'diff_weight']
    cfg = {}
    for k, t in numeric_keys.items():
        cfg[k] = t(cfg_dict.get(k, t(0)))
    for k in float_keys:
        cfg[k] = float(cfg_dict.get(k, 0.0))
    cfg['profile'] = cfg_dict.get('profile', False)
    cfg['data_dir'] = cfg_dict['data_dir']
    cfg['output_dir'] = cfg_dict['output_dir']
    return cfg

# -- Streaming Dataset ------------------------------------------------------
class StreamingTickDataset(IterableDataset):
    def __init__(self, data_dir, seq_len=100):
        super().__init__()
        self.files = sorted(Path(data_dir).glob('*.parquet'))
        self.seq_len = seq_len

    def __iter__(self):
        for file_path in self.files:
            df = load_parquet_numbers(str(file_path))
            if len(df) <= self.seq_len:
                continue
            features = df.drop('price', axis=1).values
            targets = df['price'].values
            for i in range(len(df) - self.seq_len):
                x = features[i : i + self.seq_len]
                y = targets[i + self.seq_len]
                yield torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -- Model Definition --------------------------------------------------------
class HybridModel(nn.Module):
    def __init__(self, feature_dim, d_model=64, n_heads=4, num_layers=2, diff_steps=50):
        super().__init__()
        self.patch_embed = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.forecast = nn.Linear(d_model, 1)
        self.diffusion = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, d_model)
        )
        self.diff_steps = diff_steps

    def forward(self, x):  # x: [B, T, F]
        x = self.patch_embed(x)
        h = self.transformer(x)
        h_last = h[:, -1, :]

        forecast = self.forecast(h_last).squeeze(-1)

        eps_preds = []
        diff_in = h_last
        for _ in range(self.diff_steps):
            eps = checkpoint(self.diffusion, diff_in, use_reentrant=False)
            eps_preds.append(eps)
            diff_in = diff_in - eps * 0.1
        eps_preds = torch.stack(eps_preds, dim=1)
        return forecast, eps_preds

# -- Training Loop -----------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, diff_weight):
    mse = nn.MSELoss()
    model.train()
    total_loss, fore_loss, diff_loss = 0.0, 0.0, 0.0

    for seq, target in loader:
        seq, target = seq.to(device), target.to(device)
        optimizer.zero_grad()
        forecast, eps_preds = model(seq)

        loss_f = mse(forecast, target)
        avg_eps = eps_preds.mean(dim=1)
        zeros = torch.zeros_like(avg_eps)
        loss_d = mse(avg_eps, zeros)

        loss = loss_f + diff_weight * loss_d
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        fore_loss += loss_f.item()
        diff_loss += loss_d.item()
    scheduler.step()
    return total_loss, fore_loss, diff_loss

# -- Streamlit App -----------------------------------------------------------
def main():
    st.set_page_config(page_title="Streaming Tick Training", layout="wide")
    st.title("Streaming Tick Forecast Training")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        data_dir = st.text_input('Data Directory', value='data/')
        output_dir = st.text_input('Output Directory', value='outputs/')
        seq_len = st.number_input('Sequence Length', min_value=1, value=100)
        batch_size = st.number_input('Batch Size', min_value=1, value=32)
        num_workers = st.number_input('Num Workers', min_value=0, value=4)
        d_model = st.number_input('Model Dimension', min_value=1, value=64)
        n_heads = st.number_input('Num Heads', min_value=1, value=4)
        num_layers = st.number_input('Num Layers', min_value=1, value=2)
        diff_steps = st.number_input('Diffusion Steps', min_value=1, value=50)
        lr = st.number_input('Learning Rate', value=1e-3, format="%.6f")
        gamma = st.number_input('LR Gamma', value=0.1, format="%.3f")
        step_size = st.number_input('LR Step Size', min_value=1, value=10)
        diff_weight = st.number_input('Diffusion Weight', value=1.0)
        epochs = st.number_input('Epochs', min_value=1, value=5)
        ckpt_every = st.number_input('Checkpoint Every', min_value=1, value=1)
        profile = st.checkbox('Enable Profiling')

    cfg = load_config_from_dict({
        'data_dir': data_dir, 'output_dir': output_dir,
        'seq_len': seq_len, 'batch_size': batch_size,
        'num_workers': num_workers, 'd_model': d_model,
        'n_heads': n_heads, 'num_layers': num_layers,
        'diff_steps': diff_steps, 'lr': lr, 'gamma': gamma,
        'step_size': step_size, 'diff_weight': diff_weight,
        'epochs': epochs, 'ckpt_every': ckpt_every,
        'profile': profile
    })

    if st.sidebar.button('Start Training'):
        # Prepare output
        os.makedirs(cfg['output_dir'], exist_ok=True)

        # Device selection
        device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cpu')
        )
        if device.type == 'mps':
            torch.set_float32_matmul_precision('high')

        # Dataset and loader
        ds = StreamingTickDataset(cfg['data_dir'], seq_len=cfg['seq_len'])
        pin_mem = True if device.type == 'cuda' else False
        loader = DataLoader(
            ds,
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            pin_memory=pin_mem,
            prefetch_factor=2
        )

        # Model initialization
        sample_seq, _ = next(iter(loader))
        model = HybridModel(
            feature_dim=sample_seq.shape[2],
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            num_layers=cfg['num_layers'],
            diff_steps=cfg['diff_steps']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg['step_size'], gamma=cfg['gamma']
        )

        # UI elements
        progress_bar = st.progress(0)
        chart = st.line_chart()
        log = st.empty()

        # History buffers
        hist = {'Total Loss': [], 'Forecast Loss': [], 'Diffusion Loss': []}

        # Training loop
        for epoch in range(1, cfg['epochs'] + 1):
            t_loss, f_loss, d_loss = train_one_epoch(
                model, loader, optimizer, scheduler, device, cfg['diff_weight']
            )
            # Checkpoint
            if epoch % cfg['ckpt_every'] == 0:
                ckpt_path = os.path.join(
                    cfg['output_dir'], f"model_epoch{epoch}.pt"
                )
                torch.save(model.state_dict(), ckpt_path)

            # Update history
            n_batches = len(loader)
            hist['Total Loss'].append(t_loss / n_batches)
            hist['Forecast Loss'].append(f_loss / n_batches)
            hist['Diffusion Loss'].append(d_loss / n_batches)

            # Update chart and logs
            chart.add_rows({
                'Total Loss': [hist['Total Loss'][-1]],
                'Forecast Loss': [hist['Forecast Loss'][-1]],
                'Diffusion Loss': [hist['Diffusion Loss'][-1]]
            })
            progress_bar.progress(epoch / cfg['epochs'])
            log.text(
                f"Epoch {epoch}/{cfg['epochs']} — Total: {hist['Total Loss'][-1]:.4f}, "
                f"Fore: {hist['Forecast Loss'][-1]:.4f}, Diff: {hist['Diffusion Loss'][-1]:.4f}"
            )

        # Final save
        torch.save(
            model.state_dict(), os.path.join(cfg['output_dir'], 'model_final.pt')
        )
        st.success("Training complete!")

if __name__ == '__main__':
    main()
