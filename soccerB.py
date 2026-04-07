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

    print(f"\nLoaded: {path}")
    df = pd.read_csv(path)

    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # ----------------------------
    # player_name standardization
    # ----------------------------
    if "athlete_id" in df.columns:
        df = df.rename(columns={"athlete_id": "player_name"})
    elif "player_id" in df.columns:
        df = df.rename(columns={"player_id": "player_name"})

    if "player_name" not in df.columns:
        print(f"WARNING: no player column in {name}")
        return None

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["player_name"] = df["player_name"].apply(
        lambda x: x if x.startswith("TeamB-") else f"TeamB-{x}"
    )

    # ----------------------------
    # DATE FIX (CRITICAL)
    # ----------------------------
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True).dt.date
    elif "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce", dayfirst=True).dt.date
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True).dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.date
    else:
        print(f"WARNING: no date column in {name}")
        return None

    df = df.dropna(subset=["player_name", "date"])

    # DEBUG
    print("\n--- SUBJECTIVE DEBUG ---")
    print("File:", name)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Null dates:", df["date"].isna().sum())
    print("Sample keys:")
    print(df[["player_name", "date"]].head(5))

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

print(f"\nLoaded subjective datasets: {len(subjective_dfs)}")

if len(subjective_dfs) == 0:
    subjective_all = None
else:
    subjective_all = pd.concat(subjective_dfs, ignore_index=True)

    print("\n===== SUBJECTIVE MASTER CHECK =====")
    print("Shape:", subjective_all.shape)
    print("Min date:", subjective_all["date"].min())
    print("Max date:", subjective_all["date"].max())
    print("Unique players:", subjective_all["player_name"].nunique())


# ----------------------------
# COLLECT PARQUET FILES
# ----------------------------
files = []

for root, dirs, filenames in os.walk(DATA_FOLDER):
    for f in filenames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))

files.sort()

print(f"\nFound {len(files)} parquet files")
print(f"Processing first {MAX_FILES}\n")

all_data = []
bad_files = []


# ----------------------------
# PROCESS FILE
# ----------------------------
def process_file(file_path):

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print("SKIP:", file_path, e)
        return None

    keep_cols = [
        'player_name', 'time', 'lat', 'lon', 'speed', 'heart_rate',
        'hacc', 'hdop', 'signal_quality', 'num_satellites',
        'inst_acc_impulse', 'accl_x', 'accl_y', 'accl_z',
        'gyro_x', 'gyro_y', 'gyro_z'
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

    # player_name cleanup
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str).str.strip()
        df["player_name"] = df["player_name"].apply(
            lambda x: x if x.startswith("TeamB-") else f"TeamB-{x}"
        )

    # ----------------------------
    # DATE FIX (CRITICAL)
    # ----------------------------
    filename = os.path.basename(file_path)
    date_val = pd.to_datetime(filename[:10], errors="coerce")

    df["date"] = date_val.date() if not pd.isna(date_val) else None

    # DEBUG
    print("\n--- OBJECTIVE DEBUG ---")
    print("File:", file_path)
    print("Date:", df["date"].iloc[0] if len(df) > 0 else None)
    print("Players:", df["player_name"].iloc[0] if "player_name" in df.columns else "MISSING")
    print("Rows:", len(df))

    # time parsing
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # sort
    if "player_name" in df.columns and "time" in df.columns:
        df = df.sort_values(["player_name", "time"])

    # fill numeric
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if "player_name" in df.columns:
        df[numeric_cols] = df.groupby("player_name")[numeric_cols].transform(
            lambda x: x.ffill().bfill()
        )

    # merge subjective
    if subjective_all is not None:
        df = df.merge(subjective_all, on=["player_name", "date"], how="left")

    return df


# ----------------------------
# MAIN LOOP
# ----------------------------
for i, file_path in enumerate(files[:MAX_FILES]):

    print(f"\n[{i+1}/{MAX_FILES}] {file_path}")

    df = process_file(file_path)

    if df is None:
        bad_files.append(file_path)
        continue

    all_data.append(df.head(500))

    del df
    gc.collect()


# ----------------------------
# FINAL COMBINE
# ----------------------------
print("\nCombining results...\n")

if len(all_data) > 0:

    final_df = pd.concat(all_data, ignore_index=True)

    print("SHAPE:", final_df.shape)
    print("\nSAMPLE:")
    print(final_df.head(20))

    # ----------------------------
    # KEY INTERSECTION CHECK (IMPORTANT)
    # ----------------------------
    if subjective_all is not None:

        print("\n===== KEY INTERSECTION CHECK =====")

        obj_keys = set(zip(final_df["player_name"], final_df["date"]))
        sub_keys = set(zip(subjective_all["player_name"], subjective_all["date"]))

        print("Objective keys:", len(obj_keys))
        print("Subjective keys:", len(sub_keys))
        print("Intersection:", len(obj_keys & sub_keys))

    # sample player
    player = final_df["player_name"].iloc[0]
    print("\nSample player:", player)
    print(final_df[final_df["player_name"] == player].head(20))

else:
    print("No data processed")
    final_df = None


# ----------------------------
# SUBJECTIVE COVERAGE CHECK
# ----------------------------
print("\n===== SUBJECTIVE COVERAGE =====")

if final_df is not None and subjective_all is not None:

    subjective_cols = [
        c for c in final_df.columns
        if c not in [
            'player_name','time','lat','lon','speed','heart_rate',
            'hacc','hdop','signal_quality','num_satellites',
            'inst_acc_impulse','accl_x','accl_y','accl_z',
            'gyro_x','gyro_y','gyro_z','date'
        ]
    ]

    print("Subjective columns:", subjective_cols)

    for col in subjective_cols[:10]:
        coverage = final_df[col].notna().mean()
        print(f"{col}: {coverage:.2%}")

else:
    print("Skipping coverage check (missing data)")


# ----------------------------
# SUMMARY
# ----------------------------
print("\nDONE")
print("Bad files:", len(bad_files))