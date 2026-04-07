import os
import gc
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"
SUBJECTIVE_FOLDER = "subjective"
MAX_FILES = 50

# ----------------------------
# LOAD SUBJECTIVE FILES SAFELY
# ----------------------------
def load_if_exists(path, name):
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None

    print(f"Loaded: {path}")
    df = pd.read_csv(path)

    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # ----------------------------
    # player_name handling
    # ----------------------------
    if "athlete_id" in df.columns:
        df = df.rename(columns={"athlete_id": "player_name"})
    elif "player_id" in df.columns:
        df = df.rename(columns={"player_id": "player_name"})

    if "player_name" not in df.columns:
        print(f"WARNING: no player column in {name}, skipping")
        return None

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["player_name"] = df["player_name"].apply(lambda x: x if x.startswith("TeamB-") else f"TeamB-{x}")

    # ----------------------------
    # date handling (ONLY ONCE)
    # ----------------------------
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True).dt.date
    elif "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.date
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    else:
        print(f"WARNING: no date column in {name}, skipping")
        return None

    df = df.dropna(subset=["player_name", "date"])

    print("\n--- SUBJECTIVE DEBUG ---")
    print("File:", name)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nPlayer sample:")
    print(df["player_name"].head(5))

    print("\nDate sample:")
    print(df["date"].head(5))

    print("\nNull dates:")
    print(df["date"].isna().sum(), "/", len(df))

    return df


print("Loading subjective data...")

subjective_dfs = []

paths = [
    ("injury", f"{SUBJECTIVE_FOLDER}/injury/injury.csv"),
    ("wellness", f"{SUBJECTIVE_FOLDER}/wellness/wellness.csv"),
    ("training", f"{SUBJECTIVE_FOLDER}/training-load/training-load.csv"),
    ("performance", f"{SUBJECTIVE_FOLDER}/game-performance/game-performance.csv"),
    ("illness", f"{SUBJECTIVE_FOLDER}/illness/illness.csv"),
]

for name, path in paths:
    df_temp = load_if_exists(path, name)
    if df_temp is not None:
        subjective_dfs.append(df_temp)

print(f"Loaded subjective datasets: {len(subjective_dfs)}\n")

# combine subjective data
subjective_all = pd.concat(subjective_dfs, ignore_index=True) if subjective_dfs else None

print("\n===== SUBJECTIVE MASTER TABLE CHECK =====")
print("Shape:", subjective_all.shape)

print("\nDate range:")
print("Min:", subjective_all["date"].min())
print("Max:", subjective_all["date"].max())

print("\nUnique players:")
print(subjective_all["player_name"].nunique())

print("\nSample keys:")
print(subjective_all[["player_name", "date"]].drop_duplicates().head(10))
# ----------------------------
# COLLECT PARQUET FILES
# ----------------------------
files = []

for root, dirs, filenames in os.walk(DATA_FOLDER):
    for f in filenames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))

files.sort()

print(f"Found {len(files)} parquet files")
print(f"Processing first {MAX_FILES} files...\n")

all_data = []
bad_files = []

# ----------------------------
# PROCESS FUNCTION
# ----------------------------
def process_file(file_path):

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"SKIP: {file_path}")
        print("Reason:", e)
        return None

    keep_cols = [
        'player_name', 'time', 'lat', 'lon', 'speed', 'heart_rate',
        'hacc', 'hdop', 'signal_quality', 'num_satellites',
        'inst_acc_impulse', 'accl_x', 'accl_y', 'accl_z',
        'gyro_x', 'gyro_y', 'gyro_z'
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

    # standardize player_name format
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str).str.strip()
        df["player_name"] = df["player_name"].apply(lambda x: x if x.startswith("TeamB-") else f"TeamB-{x}")

    # extract date from filename
    filename = os.path.basename(file_path)
    date_val = pd.to_datetime(filename[:10], errors="coerce")
    df["date"] = date_val.date() if not pd.isna(date_val) else None

    print("\n--- OBJECTIVE DEBUG ---")
    print("File:", file_path)

    print("Date:", df["date"].iloc[0] if len(df) > 0 else None)

    print("Player sample:", df["player_name"].iloc[0] if "player_name" in df.columns else "MISSING")

    print("Rows:", len(df))

    # parse time safely
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", format="mixed")

    # sort data
    if "player_name" in df.columns and "time" in df.columns:
        df = df.sort_values(["player_name", "time"])

    # impute missing numeric values
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if "player_name" in df.columns and len(numeric_cols) > 0:
        df[numeric_cols] = df.groupby("player_name")[numeric_cols].transform(lambda x: x.ffill().bfill())

    # merge subjective data
    if subjective_all is not None:
        df = df.merge(subjective_all, on=["player_name", "date"], how="left")

    return df


# ----------------------------
# MAIN LOOP
# ----------------------------
for i, file_path in enumerate(files[:MAX_FILES]):

    print(f"[{i+1}/{MAX_FILES}] {file_path}")

    df = process_file(file_path)

    if df is None:
        bad_files.append(file_path)
        continue

    all_data.append(df.head(500))

    del df
    gc.collect()


# ----------------------------
# COMBINE RESULTS
# ----------------------------
print("\nCombining results...\n")

if len(all_data) > 0:
    final_df = pd.concat(all_data, ignore_index=True)

    print("===== FINAL DATA SAMPLE =====")
    print(final_df.head(30))

    print("\nSHAPE:", final_df.shape)
    print("\nCOLUMNS:", list(final_df.columns))

    # merge check
    if subjective_all is not None:
        print("\n===== MERGE CHECK =====")
        print(final_df[['player_name', 'date']].drop_duplicates().head())

    # show one athlete
    if "player_name" in final_df.columns:
        sample_player = final_df["player_name"].iloc[0]
        player_df = final_df[final_df["player_name"] == sample_player]

        print("\n===== ONE ATHLETE OVER TIME =====")
        print("Athlete:", sample_player)
        print(player_df.head(30))

else:
    print("No valid data processed")

# ----------------------------
# SUBJECTIVE VALIDATION
# ----------------------------
print("\n===== SUBJECTIVE COLUMNS AFTER MERGE =====")
base_cols = [
    'player_name','time','lat','lon','speed','heart_rate',
    'hacc','hdop','signal_quality','num_satellites',
    'inst_acc_impulse','accl_x','accl_y','accl_z',
    'gyro_x','gyro_y','gyro_z','date'
]
subjective_cols = [c for c in final_df.columns if c not in base_cols]
print(subjective_cols)

print("\n===== SUBJECTIVE MERGE COVERAGE =====")
for col in subjective_cols[:10]:
    non_null = final_df[col].notna().sum()
    total = len(final_df)
    print(f"{col}: {non_null}/{total} ({non_null/total:.2%})")

print("\n===== MERGE SUCCESS CHECK =====")

merged_cols = ["player_name", "date"] + [c for c in df.columns if c not in ["player_name", "time", "lat", "lon", "speed"]]

print("Null subjective rows:", df[subjective_all.columns].isna().all(axis=1).mean())

print("\nCoverage per column:")
for col in subjective_all.columns:
    if col in df.columns:
        print(col, "->", df[col].notna().mean())

print("\n===== MERGE SUCCESS CHECK =====")

merged_cols = ["player_name", "date"] + [c for c in df.columns if c not in ["player_name", "time", "lat", "lon", "speed"]]

print("Null subjective rows:", df[subjective_all.columns].isna().all(axis=1).mean())

print("\nCoverage per column:")
for col in subjective_all.columns:
    if col in df.columns:
        print(col, "->", df[col].notna().mean())

# ----------------------------
# SUMMARY
# ----------------------------
print("\nDONE")
print("Bad files:", len(bad_files))