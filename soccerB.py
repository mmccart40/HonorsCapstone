import os
import gc
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"
MAX_FILES = 50  # safety limit for memory

# ----------------------------
# COLLECT FILES
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

    # ------------------------
    # SAFE PARQUET LOAD
    # ------------------------
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"SKIP (corrupt/non-parquet): {file_path}")
        print("Reason:", e)
        return None

    # ------------------------
    # COLUMN FILTER (safe)
    # ------------------------
    keep_cols = [
        'player_name', 'time', 'lat', 'lon', 'speed', 'heart_rate',
        'hacc', 'hdop', 'signal_quality', 'num_satellites',
        'inst_acc_impulse', 'accl_x', 'accl_y', 'accl_z',
        'gyro_x', 'gyro_y', 'gyro_z'
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

    # ------------------------
    # ADD DATE FROM FILE NAME (FAST)
    # ------------------------
    filename = os.path.basename(file_path)
    df["date"] = filename[:10]

    # ------------------------
    # SORT BY ATHLETE + TIME
    # ------------------------
    if "player_name" in df.columns and "time" in df.columns:
        df = df.sort_values(["player_name", "time"])

    # ------------------------
    # IMPUTE MISSING VALUES PER ATHLETE
    # ------------------------
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    if "player_name" in df.columns and len(numeric_cols) > 0:
        df[numeric_cols] = (
            df.groupby("player_name")[numeric_cols]
            .transform(lambda x: x.ffill().bfill())
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

    # keep memory small (sample per file)
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
    # SHOW ONE ATHLETE TIME SERIES
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