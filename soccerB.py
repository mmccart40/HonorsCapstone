import os
import gc
import pandas as pd
import numpy as np

# ----------------------------
# ROOT DATA PATH
# ----------------------------
DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"

# ----------------------------
# COLLECT FILES
# ----------------------------
files = []

for root, dirs, filenames in os.walk(DATA_FOLDER):
    for f in filenames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))

print(f"Found {len(files)} parquet files")

if len(files) == 0:
    print("No files found")
    exit()

# ----------------------------
# LOAD ALL FILES
# ----------------------------
dfs = []

for i, file_path in enumerate(files):
    try:
        df = pd.read_parquet(file_path)
        df["source_file"] = os.path.basename(file_path)
        dfs.append(df)
    except Exception as e:
        print("Skipping file:", file_path, e)

df = pd.concat(dfs, ignore_index=True)

print("\nRaw shape:", df.shape)

# ----------------------------
# CLEAN COLUMN NAMES
# ----------------------------
df.columns = (
    df.columns.str.lower()
    .str.replace(" ", "_")
    .str.replace(".", "", regex=False)
)

# ----------------------------
# FIX TIME COLUMN
# ----------------------------
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"])

# ----------------------------
# SORT DATA (CRITICAL)
# ----------------------------
df = df.sort_values(["player_name", "time"]).reset_index(drop=True)

print("\nAfter sorting:", df.shape)

# ----------------------------
# IMPUTATION PER PLAYER
# ----------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

filled_dfs = []

for player in df["player_name"].unique():

    player_df = df[df["player_name"] == player].copy()
    player_df = player_df.sort_values("time")

    # forward fill + backward fill
    player_df[num_cols] = player_df[num_cols].ffill().bfill()

    # final fallback: column mean
    player_df[num_cols] = player_df[num_cols].fillna(player_df[num_cols].mean())

    filled_dfs.append(player_df)

df_clean = pd.concat(filled_dfs, ignore_index=True)

# final safety sort
df_clean = df_clean.sort_values(["player_name", "time"]).reset_index(drop=True)

# ----------------------------
# SAMPLE OUTPUT (YOU ASKED FOR THIS)
# ----------------------------
print("\n===== SAMPLE DATA =====")
print(df_clean.head(30))

print("\nCOLUMNS:")
print(df_clean.columns)

print("\nSHAPE:")
print(df_clean.shape)

print("\nNULL CHECK (top 10):")
print(df_clean.isnull().sum().sort_values(ascending=False).head(10))

# ----------------------------
# OPTIONAL: SIMPLE DAILY AGGREGATION (FOR BASELINE MODELS)
# ----------------------------
df_clean["date"] = df_clean["time"].dt.date

daily_df = df_clean.groupby(["player_name", "date"]).agg({
    "speed": "mean",
    "heart_rate": "mean",
    "inst_acc_impulse": "sum",
    "accl_x": "std",
    "accl_y": "std",
    "accl_z": "std"
}).reset_index()

print("\n===== DAILY AGGREGATED SAMPLE =====")
print(daily_df.head(20))

print("\nDAILY SHAPE:", daily_df.shape)

# ----------------------------
# SAVE CLEAN DATASETS
# ----------------------------
df_clean.to_parquet("soccerB_timeseries_clean.parquet", index=False)
daily_df.to_parquet("soccerB_daily_features.parquet", index=False)

print("\nSaved:")
print("- soccerB_timeseries_clean.parquet")
print("- soccerB_daily_features.parquet")

# ----------------------------
# CLEAN MEMORY
# ----------------------------
gc.collect()

print("\nDONE")