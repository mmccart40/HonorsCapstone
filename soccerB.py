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
    if os.path.exists(path):
        print(f"Loaded: {path}")
        df = pd.read_csv(path)

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(" ", "_")

        # Ensure player + date columns exist
        if "player_name" not in df.columns:
            if "athlete_id" in df.columns:
                df = df.rename(columns={"athlete_id": "player_name"})

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        df["source"] = name
        return df
    else:
        print(f"Missing: {path}")
        return None


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

# Combine subjective data
if subjective_dfs:
    subjective_all = pd.concat(subjective_dfs, ignore_index=True)
else:
    subjective_all = None

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

    # ------------------------
    # FIX DATE PARSING (NO .dt ERROR)
    # ------------------------
    filename = os.path.basename(file_path)

    try:
        date_val = pd.to_datetime(filename[:10], errors="coerce")
        df["date"] = date_val.date() if pd.notna(date_val) else None
    except:
        df["date"] = None

    # ------------------------
    # SORT
    # ------------------------
    if "player_name" in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values(["player_name", "time"])

    # ------------------------
    # IMPUTE
    # ------------------------
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    if "player_name" in df.columns and len(numeric_cols) > 0:
        df[numeric_cols] = (
            df.groupby("player_name")[numeric_cols]
            .transform(lambda x: x.ffill().bfill())
        )

    # ------------------------
    # MERGE SUBJECTIVE DATA
    # ------------------------
    if subjective_all is not None:
        if "player_name" in df.columns and "date" in df.columns:
            df = df.merge(
                subjective_all,
                on=["player_name", "date"],
                how="left"
            )

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

    # ----------------------------
    # DEBUG: CHECK MERGE SUCCESS
    # ----------------------------
    if subjective_all is not None:
        print("\n===== MERGE CHECK =====")
        print(final_df[['player_name', 'date']].drop_duplicates().head())

    # ----------------------------
    # SHOW ONE ATHLETE
    # ----------------------------
    if "player_name" in final_df.columns:
        sample_player = final_df["player_name"].iloc[0]
        player_df = final_df[final_df["player_name"] == sample_player]

        print("\n===== ONE ATHLETE OVER TIME =====")
        print("Athlete:", sample_player)
        print(player_df.head(30))

else:
    print("No valid data processed")

# ----------------------------
# SUMMARY
# ----------------------------
print("\nDONE")
print("Bad files:", len(bad_files))