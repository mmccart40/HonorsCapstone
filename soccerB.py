import os
import gc
import pandas as pd
import pyarrow as pa

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

# ----------------------------
# DISPLAY ONE FILE STRUCTURE
# ----------------------------
if len(files) > 0:
    print("\n--- ONE FILE STRUCTURE PREVIEW ---\n")
    try:
        sample_df = pd.read_parquet(files[0])
        print("COLUMNS:", sample_df.columns)
        print("\nHEAD:\n", sample_df.head())
        print("\nDTYPES:\n", sample_df.dtypes)
    except Exception as e:
        print("Could not preview first file:", e)

# ----------------------------
# BUILD MAP OF DATES → FILES
# ----------------------------
day_map = {}

for f in files:
    try:
        base = os.path.basename(f)
        date = base.split("-TeamB")[0]  # e.g., 2020-07-21
        if date not in day_map:
            day_map[date] = []
        day_map[date].append(f)
    except:
        pass

# pick the first available day
if len(day_map) == 0:
    print("No valid day folders found")
    sample_day = None
    day_files = []
else:
    sample_day = list(day_map.keys())[0]
    day_files = day_map[sample_day]

print(f"\n--- LOADING ONE FULL DAY: {sample_day} ---")
print(f"Files for this day: {len(day_files)}")

# ----------------------------
# LOAD ALL FILES FOR THAT DAY
# ----------------------------
day_dfs = []
for f in day_files:
    try:
        df = pd.read_parquet(f)
        day_dfs.append(df)
    except Exception as e:
        print("Skip file in day load:", f, e)

if len(day_dfs) > 0:
    day_data = pd.concat(day_dfs, ignore_index=True)
    print("\n--- ONE DAY SAMPLE TABLE (first 20 rows) ---\n")
    print(day_data.head(20))
    print("\nSHAPE:", day_data.shape)
    print("\nCOLUMNS:", day_data.columns)
else:
    print("No data loaded for sample day")

# ----------------------------
# YOUR ORIGINAL PROCESS PIPELINE
# ----------------------------
def process(df):
    if "athlete_id" in df.columns:
        return df.groupby("athlete_id").size()
    return None

results = []
bad_files = []

for i, file_path in enumerate(files):
    print(f"\n[{i+1}/{len(files)}] {file_path}")
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
# FINAL COMBINE
# ----------------------------
print("\nCombining results...")
if results:
    final_result = pd.concat(results)
    print(final_result.head())
else:
    print("No results generated")

print("\nBad files:", len(bad_files))
print("DONE")