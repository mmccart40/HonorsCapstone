import os
import gc
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"
MAX_FILES = 50   # 🔥 limit to avoid OOM (increase later)

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

bad_files = []
all_data = []

# ----------------------------
# PROCESS FUNCTION
# ----------------------------
def process_file(file_path):
    df = pd.read_parquet(file_path)

    # Keep only relevant columns
    keep_cols = [
        'player_name', 'time', 'lat', 'lon', 'speed', 'heart_rate',
        'hacc', 'hdop', 'signal_quality', 'num_satellites',
        'inst_acc_impulse', 'accl_x', 'accl_y', 'accl_z',
        'gyro_x', 'gyro_y', 'gyro_z'
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Extract date from filename (FAST, avoids warnings)
    filename = os.path.basename(file_path)
    df["date"] = filename[:10]

    # Sort by player and time
    df = df.sort_values(["player_name", "time"])

    # Impute missing values per athlete
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

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

    try:
        df = process_file(file_path)

        # Keep a SMALL sample from each file to avoid memory explosion
        all_data.append(df.head(500))

    except Exception as e:
        print("ERROR:", file_path)
        print(e)
        bad_files.append(file_path)

    del df
    gc.collect()


# ----------------------------
# COMBINE SAMPLE DATA
# ----------------------------
print("\nCombining sampled data...\n")

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    print("===== FINAL SAMPLE =====")
    print(final_df.head(30))

    print("\nSHAPE:", final_df.shape)

    print("\nCOLUMNS:")
    print(final_df.columns)

    # ----------------------------
    # SHOW ONE ATHLETE OVER TIME
    # ----------------------------
    sample_player = final_df["player_name"].iloc[0]

    player_df = final_df[final_df["player_name"] == sample_player]

    print(f"\n===== TIME SERIES FOR ONE ATHLETE =====")
    print(f"Athlete: {sample_player}\n")
    print(player_df.head(30))

else:
    print("No data processed")

print("\nBad files:", len(bad_files))
print("DONE")