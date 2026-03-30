import os
import gc
import pandas as pd

# ----------------------------
# ROOT DATA PATH
# ----------------------------
DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"


# ----------------------------
# RECURSIVE FILE COLLECTION (CRITICAL FIX)
# ----------------------------
files = []

for root, dirs, filenames in os.walk(DATA_FOLDER):
    for f in filenames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))

print(f"Found {len(files)} parquet files")


# ----------------------------
# PROCESS FUNCTION (REPLACE WITH YOUR MODEL)
# ----------------------------
def process(df):
    print("Shape:", df.shape)

    # Example lightweight aggregation
    if "athlete_id" in df.columns:
        return df.groupby("athlete_id").size()

    return None


results = []


# ----------------------------
# MAIN LOOP (OOM SAFE)
# ----------------------------
for i, file_path in enumerate(files):
    print(f"\n[{i+1}/{len(files)}] {file_path}")

    df = pd.read_parquet(file_path)

    result = process(df)

    if result is not None:
        results.append(result)

    # MEMORY CLEANUP (CRITICAL)
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

print("DONE")