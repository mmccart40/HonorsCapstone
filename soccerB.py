import os
import gc
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"
SUBJECTIVE_FOLDER = "subjective"
MAX_FILES = 50
ROWS_PER_FILE = 200   # keep small to avoid OOM

# ----------------------------
# HELPERS
# ----------------------------
def standardize_player_name(x):
    x = str(x).strip()
    # FIX: convert TeamA -> TeamB so merge works
    if x.startswith("TeamA-"):
        return x.replace("TeamA-", "TeamB-")
    if not x.startswith("TeamB-"):
        return f"TeamB-{x}"
    return x


def parse_date_strict(series):
    # FIX: no warnings, consistent parsing
    return pd.to_datetime(series, format="%d.%m.%Y", errors="coerce").dt.date


# ----------------------------
# LOAD SUBJECTIVE DATA
# ----------------------------
def load_subjective_file(name, path):
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None

    print(f"\nLoading subjective: {name}")
    df = pd.read_csv(path)

    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    print("Raw shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if "player_name" not in df.columns:
        print("Skipping: no player_name")
        return None

    df["player_name"] = df["player_name"].apply(standardize_player_name)

    # FIX: strict timestamp parsing (no warnings)
    if "timestamp" in df.columns:
        df["date"] = parse_date_strict(df["timestamp"])
    else:
        print("Skipping: no timestamp column")
        return None

    before = len(df)
    df = df.dropna(subset=["player_name", "date"])
    after = len(df)

    print(f"Rows after cleaning: {after} (dropped {before-after})")

    return df


print("Loading subjective data...")

paths = [
    ("injury", f"{SUBJECTIVE_FOLDER}/injury/injury.csv"),
    ("wellness", f"{SUBJECTIVE_FOLDER}/wellness/wellness.csv"),
    ("training", f"{SUBJECTIVE_FOLDER}/training-load/training-load.csv"),
    ("performance", f"{SUBJECTIVE_FOLDER}/game-performance/game-performance.csv"),
    ("illness", f"{SUBJECTIVE_FOLDER}/illness/illness.csv"),
]

subjective_dfs = []

for name, path in paths:
    df_temp = load_subjective_file(name, path)
    if df_temp is not None:
        subjective_dfs.append(df_temp)

if len(subjective_dfs) > 0:
    subjective_all = pd.concat(subjective_dfs, ignore_index=True)
else:
    subjective_all = None

if subjective_all is not None:
    print("\nSubjective rows:", len(subjective_all))
    print("Unique players:", subjective_all["player_name"].nunique())
    print("Date range:",
          subjective_all["date"].min(),
          "to",
          subjective_all["date"].max())
else:
    print("No subjective data loaded")


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
# PROCESS OBJECTIVE FILE
# ----------------------------
def process_file(file_path):
    try:
        df = pd.read_parquet(file_path)
    except Exception:
        print("Skipping file:", file_path)
        return None

    # KEEP ONLY ESSENTIAL COLUMNS (memory safe)
    keep_cols = [
        "player_name", "time",
        "speed", "heart_rate",
        "accl_x", "accl_y", "accl_z"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    if "player_name" not in df.columns:
        return None

    df["player_name"] = df["player_name"].apply(standardize_player_name)

    # extract date from filename
    filename = os.path.basename(file_path)
    date_val = pd.to_datetime(filename[:10], errors="coerce")
    if pd.isna(date_val):
        return None

    df["date"] = date_val.date()

    # LIMIT ROWS EARLY (critical for memory)
    df = df.head(ROWS_PER_FILE)

    # OPTIONAL: reduce memory further
    for col in ["speed", "heart_rate", "accl_x", "accl_y", "accl_z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    # MERGE SUBJECTIVE
    if subjective_all is not None:
        df = df.merge(subjective_all, on=["player_name", "date"], how="left")

    return df


# ----------------------------
# MAIN LOOP
# ----------------------------
all_data = []
bad_files = []

for i, file_path in enumerate(files[:MAX_FILES]):
    print(f"[{i+1}/{MAX_FILES}] Processing")

    df_part = process_file(file_path)

    if df_part is None:
        bad_files.append(file_path)
        continue

    all_data.append(df_part)

    del df_part
    gc.collect()


# ----------------------------
# COMBINE
# ----------------------------
print("\nCombining...")

if len(all_data) > 0:
    final_df = pd.concat(all_data, ignore_index=True)

    print("Final shape:", final_df.shape)
    print("Columns:", list(final_df.columns))

    # FIX: no more NameError (use final_df, not df)
    print("\nOBJECTIVE DATE RANGE:",
          final_df["date"].min(),
          "to",
          final_df["date"].max())

    # ----------------------------
    # MERGE CHECK
    # ----------------------------
    if subjective_all is not None:
        print("\nMerge coverage:")
        coverage = final_df[subjective_all.columns].notna().mean()
        print(coverage)

        print("\nSample merged rows:")
        print(final_df.dropna(subset=subjective_all.columns).head())

else:
    print("No data processed")


# ----------------------------
# SUMMARY
# ----------------------------
print("\nDone")
print("Bad files:", len(bad_files))