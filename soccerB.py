import os
import gc
import pandas as pd

# ----------------------------
# ROOT PATH
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
# PROCESS FUNCTION (FIXED)
# ----------------------------
def process(df):
    if "player_name" not in df.columns:
        return None

    return df.groupby("player_name").agg({
        "speed": "mean",
        "heart_rate": "mean",
        "inst_acc_impulse": "mean",
        "lat": "count"   # number of records
    }).rename(columns={"lat": "num_samples"})

# ----------------------------
# MAIN LOOP
# ----------------------------
results = []
bad_files = []

for i, file_path in enumerate(files):
    print(f"[{i+1}/{len(files)}] {file_path}")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print("SKIP FILE:", file_path, e)
        bad_files.append(file_path)
        continue

    try:
        result = process(df)
        if result is not None:
            results.append(result)

    except Exception as e:
        print("Processing error:", file_path, e)

    del df
    gc.collect()

# ----------------------------
# FINAL OUTPUT
# ----------------------------
print("\nCombining results...")

if results:
    final_result = pd.concat(results)

    # reset index for readability
    final_result = final_result.reset_index()

    print("\n===== PLAYER-LEVEL SAMPLE =====")
    print(final_result.head(20))

    print("\nSHAPE:", final_result.shape)

else:
    print("No results generated")

print("\nBad files:", len(bad_files))
print("DONE")