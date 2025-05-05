import pandas as pd
from pathlib import Path

labels_dir = Path('labels')
dfs = [pd.read_csv(p) for p in labels_dir.glob('*_labels.csv')]
df = pd.concat(dfs, ignore_index=True)

print(df.label.value_counts(normalize=True))