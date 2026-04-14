import os
import gc
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamA-2020"
SUBJECTIVE_FOLDER = "subjective"
MAX_FILES = 50

# ----------------------------
# CLEAN PLAYER NAME
# ----------------------------
def clean_player_name(x):
    return str(x).strip()

# ----------------------------
# LOAD SUBJECTIVE DATA
# ----------------------------
def load_subjective(path, name):
    if not os.path.exists(path):
        print("Missing:", path)
        return None

    print("\nLoading subjective:", name)

    df = pd.read_csv(path)
    print("Raw shape:", df.shape)

    df.columns = df.columns.str.lower().str.strip()

    if "player_name" not in df.columns:
        return None

    df["player_name"] = df["player_name"].apply(clean_player_name)

    # FIXED: explicit date format (no warnings)
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(
            df["timestamp"],
            format="%d.%m.%Y",
            errors="coerce"
        )
    else:
        return None

    before = len(df)
    df = df.dropna(subset=["player_name", "date"])
    print(f"Rows after cleaning: {len(df)} (dropped {before - len(df)})")

    df = df.sort_values(["player_name", "date"])

    return df

# ----------------------------
# LOAD ALL SUBJECTIVE FILES
# ----------------------------
print("Loading subjective data...")

paths = [
    ("injury", "subjective/injury/injury.csv"),
    ("performance", "subjective/game-performance/game-performance.csv"),
    ("illness", "subjective/illness/illness.csv"),
    ("wellness", "subjective/wellness/wellness.csv"),
    ("training", "subjective/training-load/training-load.csv"),
]

subjective_dfs = []

for name, path in paths:
    df = load_subjective(path, name)
    if df is not None:
        subjective_dfs.append(df)

if len(subjective_dfs) == 0:
    print("No subjective data loaded")
    exit()

subjective_all = pd.concat(subjective_dfs, ignore_index=True)

print("\nSubjective rows:", len(subjective_all))
print("Unique players:", subjective_all["player_name"].nunique())
print("Date range:", subjective_all["date"].min(), "to", subjective_all["date"].max())

# ----------------------------
# PROCESS OBJECTIVE FILE
# ----------------------------
def process_file(file_path):

    try:
        df = pd.read_parquet(file_path)
    except:
        print("Skipping file:", file_path)
        return None

    keep_cols = [
        "player_name", "time", "speed",
        "heart_rate", "accl_x", "accl_y", "accl_z"
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

    if "player_name" not in df.columns:
        return None

    df["player_name"] = df["player_name"].apply(clean_player_name)

    # Extract date from filename
    filename = os.path.basename(file_path)
    date_val = pd.to_datetime(
        filename[:10],
        format="%Y-%m-%d",
        errors="coerce"
    )

    if pd.isna(date_val):
        return None

    df["date"] = date_val

    # Reduce size: aggregate per player-day
    df = df.groupby(["player_name", "date"]).agg({
        "speed": ["mean", "max"],
        "heart_rate": ["mean", "max"],
        "accl_x": "mean",
        "accl_y": "mean",
        "accl_z": "mean"
    }).reset_index()

    df.columns = ["_".join(col).strip("_") for col in df.columns]

    return df

# ----------------------------
# COLLECT PARQUET FILES
# ----------------------------
files = []

for root, _, filenames in os.walk(DATA_FOLDER):
    for f in filenames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))

files.sort()

print("\nTotal parquet files:", len(files))

# ----------------------------
# PROCESS OBJECTIVE FILES
# ----------------------------
all_data = []
bad_files = []

for i, file_path in enumerate(files[:MAX_FILES]):
    print(f"[{i+1}/{MAX_FILES}] Processing")

    df = process_file(file_path)

    if df is None:
        bad_files.append(file_path)
        continue

    all_data.append(df)

    del df
    gc.collect()

if len(all_data) == 0:
    print("No objective data processed")
    exit()

final_df = pd.concat(all_data, ignore_index=True)

print("\nObjective date range:",
      final_df["date"].min(),
      "to",
      final_df["date"].max())

# ----------------------------
# FIX: MERGE USING NEAREST DATE
# ----------------------------
final_df = final_df.sort_values(["player_name", "date"])
subjective_all = subjective_all.sort_values(["player_name", "date"])

merged = pd.merge_asof(
    final_df,
    subjective_all,
    on="date",
    by="player_name",
    direction="backward"
)

# ----------------------------
# RESULTS
# ----------------------------
print("\nFinal shape:", merged.shape)

print("\nMerge coverage:")
print(merged.notna().mean())

print("\nSample merged rows:")
print(merged.dropna().head())

print("\nDone")
print("Bad files:", len(bad_files))