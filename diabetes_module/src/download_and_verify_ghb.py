import os
import urllib.request
from pathlib import Path
import pandas as pd

urls = {
    'GHB_H.xpt': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/GHB_H.xpt',
    'GHB_I.xpt': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GHB_I.xpt',
    'GHB_J.xpt': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GHB_J.xpt'
}

dest_dir = Path('data/raw/nhanes')
dest_dir.mkdir(parents=True, exist_ok=True)

for fname, url in urls.items():
    dest_path = dest_dir / fname
    if not dest_path.exists():
        print(f"Downloading {fname} from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as out_f:
            out_f.write(resp.read())
        print(f"Downloaded {fname}: {dest_path.stat().st_size:,} bytes")
    else:
        print(f"{fname} already exists: {dest_path.stat().st_size:,} bytes")

print("\n--- Verifying read_sas & LBXGH ---")
for fname in sorted(urls.keys()):
    dest_path = dest_dir / fname
    df = pd.read_sas(dest_path)
    has_lbxgh = 'LBXGH' in df.columns
    has_seqn = 'SEQN' in df.columns
    print(f"{fname}: shape={df.shape}, LBXGH present={has_lbxgh}, SEQN present={has_seqn}")
    if has_lbxgh:
        valid_cnt = df['LBXGH'].notna().sum()
        min_v = df['LBXGH'].min()
        max_v = df['LBXGH'].max()
        print(f"  LBXGH non-null: {valid_cnt:,} / {len(df):,}, range: [{min_v:.2f}, {max_v:.2f}]")
