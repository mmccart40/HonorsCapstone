import os
import gc
import pandas as pd
import pyarrow as pa

# ----------------------------
# ROOT DATA PATH (FIXED)
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
# QUICK DATA INSPECTION (NEW ADDITION)
# ----------------------------
if len(files) > 0:
    print("\nInspecting first file for structure...\n")

    try:
        sample_df = pd.read_parquet(files[0])

        print("COLUMNS:")
        print(sample_df.columns)

        print("\nSHAPE:")
        print(sample_df.shape)

        print("\nHEAD:")
        print(sample_df.head())

        print("\nDTYPES:")
        print(sample_df.dtypes)

    except Exception as e:
        print("Could not inspect sample file:")
        print(e)


# ----------------------------
# PROCESS FUNCTION (PLACEHOLDER)
# ----------------------------
def process(df):

    # DEBUG: show columns occasionally (helps confirm schema consistency)
    # print(df.columns)

    if "athlete_id" in df.columns:
        return df.groupby("athlete_id").size()

    return None


results = []
bad_files = []


# ----------------------------
# MAIN LOOP (ROBUST)
# ----------------------------
for i, file_path in enumerate(files):

    print(f"\n[{i+1}/{len(files)}] {file_path}")

    try:
        df = pd.read_parquet(file_path)

    except Exception as e:
        print("SKIP FILE (corrupt or invalid parquet):", file_path)
        print("Reason:", e)
        bad_files.append(file_path)
        continue

    try:
        result = process(df)

        if result is not None:
            results.append(result)

    except Exception as e:
        print("Processing error:", file_path)
        print(e)

    del df
    gc.collect()


# ----------------------------
# FINAL COMBINE
# ----------------------------
print("\nCombining results...")

if results:
    final_result = pd.concat(results)

    print("\nFINAL RESULT PREVIEW:")
    print(final_result.head())

else:
    print("No results generated")

print("\nBad files:", len(bad_files))

print("DONE")