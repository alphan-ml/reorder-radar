import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from reorder_radar.model import FEATURE_COLS, MODEL_PATH

ROOT = Path('.')
OUT = ROOT / 'aws-lambda' / 'artifacts'
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet('outputs/holdout_features.parquet')
df = df.sort_values(['user_id', 'product_id']).reset_index(drop=True)

user_ids = df['user_id'].to_numpy(dtype='int32')
product_ids = df['product_id'].to_numpy(dtype='int32')
feat = df[FEATURE_COLS].to_numpy(dtype='float32')

# build index: user_id -> [start, count], rows already sorted by user_id
index = {}
uniq_users, starts, counts = np.unique(user_ids, return_index=True, return_counts=True)
for u, s, c in zip(uniq_users.tolist(), starts.tolist(), counts.tolist()):
    index[str(u)] = [s, c]

np.savez_compressed(OUT / 'holdout_features.npz', user_id=user_ids, product_id=product_ids, features=feat)
with open(OUT / 'holdout_index.json', 'w') as f:
    json.dump({'feature_cols': FEATURE_COLS, 'index': index}, f)

# product name lookup
prod = pd.read_csv('data/raw/products.csv')
prod_map = dict(zip(prod['product_id'].astype(int), prod['product_name'].astype(str)))
with open(OUT / 'products.json', 'w') as f:
    json.dump(prod_map, f)

# model + train info
import shutil
shutil.copy(MODEL_PATH, OUT / 'lambdarank_model.txt')
shutil.copy('outputs/train_info.json', OUT / 'train_info.json')

print('rows', len(df), 'users', len(index), 'products', len(prod_map))
print('feature_cols', FEATURE_COLS)
