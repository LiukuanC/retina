"""Build eval_val.zip (folders 0-4) from the SAME validation split used in train_v2.py.
Original image files are copied untouched (no resizing / re-compression), so the app's
numbers match the training log. Run from the folder that contains kaggle_retina/."""
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT, SEED = Path("kaggle_retina"), 5566

df = pd.read_csv(ROOT / "train.csv")
_, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["diagnosis"])

with zipfile.ZipFile("eval_val.zip", "w", zipfile.ZIP_STORED) as z:
    for _, r in val_df.iterrows():
        z.write(ROOT / "train" / f"{r['id_code']}.png", f"{int(r['diagnosis'])}/{r['id_code']}.png")
print(f"Wrote eval_val.zip with {len(val_df)} images")
print(val_df["diagnosis"].value_counts().sort_index().to_string())
