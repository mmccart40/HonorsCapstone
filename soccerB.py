import os
import gc
import pandas as pd
import numpy as np

DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"

output_file = "soccerB_timeseries_clean.parquet"

files = []
for root, dirs, filenames in os.walk(DATA_FOLDER):
    for f in filenames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))

print(f"Found {len(files)} parquet files")

bad_files = []
first_write = True

# ----------------------------
# PROCESS EACH FILE
# ----------------------------
for i, file_path in enumerate(files):

    print(f"\n[{i+1}/{len(files)}] {file_path}")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print("Skipping:", file_path)
        bad_files.append(file_path)
        continue

    # ----------------------------
    # CLEAN
    # ----------------------------
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )

    # time fix
    if "time" not in df.columns:
        continue

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    # sort within file
    df = df.sort_values(["player_name", "time"])

    # ----------------------------
    # IMPUTE (PER FILE, APPROX)
    # ----------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns

    df[num_cols] = df[num_cols].ffill().bfill()
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # ----------------------------
    # SAVE INCREMENTALLY
    # ----------------------------
    if first_write:
        df.to_parquet(output_file, index=False)
        first_write = False
    else:
        df.to_parquet(output_file, index=False, append=True)

    # ----------------------------
    # SAMPLE PRINT (FIRST FILE ONLY)
    # ----------------------------
    if i == 0:
        print("\n===== SAMPLE =====")
        print(df.head(20))
        print("\nColumns:", df.columns)

    # cleanup
    del df
    gc.collect()

# ----------------------------
# FINAL INFO
# ----------------------------
print("\nDONE PROCESSING")
print("Bad files:", len(bad_files))

print("\nLoading small sample for display...")

sample_df = pd.read_parquet(output_file).head(50)

print("\n===== FINAL SAMPLE =====")
print(sample_df)

print("\nShape (sample):", sample_df.shape)