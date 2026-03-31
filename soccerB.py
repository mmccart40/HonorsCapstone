import os
import gc
import pandas as pd
import numpy as np

DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"
CHUNK_DIR = "soccer_chunks"

os.makedirs(CHUNK_DIR, exist_ok=True)

files = []
for root, dirs, filenames in os.walk(DATA_FOLDER):
    for f in filenames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))

print(f"Found {len(files)} parquet files")

bad_files = []

# ----------------------------
# PROCESS FILES ONE BY ONE
# ----------------------------
for i, file_path in enumerate(files):

    print(f"\n[{i+1}/{len(files)}] {file_path}")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print("Skipping:", file_path)
        bad_files.append(file_path)
        continue

    # ----------------------------
    # CLEAN COLUMNS
    # ----------------------------
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )

    if "time" not in df.columns:
        continue

    # ----------------------------
    # TIME + SORT
    # ----------------------------
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    df = df.sort_values(["player_name", "time"])

    # ----------------------------
    # IMPUTE
    # ----------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns

    df[num_cols] = df[num_cols].ffill().bfill()
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # ----------------------------
    # SAVE CHUNK
    # ----------------------------
    chunk_path = os.path.join(CHUNK_DIR, f"part_{i}.parquet")
    df.to_parquet(chunk_path, index=False)

    # ----------------------------
    # SHOW SAMPLE ONCE
    # ----------------------------
    if i == 0:
        print("\n===== SAMPLE =====")
        print(df.head(20))
        print("\nColumns:", df.columns)

    # cleanup
    del df
    gc.collect()

# ----------------------------
# FINAL
# ----------------------------
print("\nDONE PROCESSING")
print("Bad files:", len(bad_files))

# ----------------------------
# LOAD SMALL SAMPLE
# ----------------------------
print("\nLoading sample from chunks...")

sample_files = os.listdir(CHUNK_DIR)[:5]
sample_dfs = [pd.read_parquet(os.path.join(CHUNK_DIR, f)) for f in sample_files]

sample_df = pd.concat(sample_dfs).head(50)

print("\n===== FINAL SAMPLE =====")
print(sample_df)

print("\nShape (sample):", sample_df.shape)