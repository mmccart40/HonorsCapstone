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
# STEP 1: INSPECT ONE FILE
# ----------------------------
print("\n===== SAMPLE FILE INSPECTION =====")

sample = pd.read_parquet(files[0])

print("\nCOLUMNS:")
print(sample.columns)

print("\nFIRST 5 ROWS:")
print(sample.head())

print("\nDTYPES:")
print(sample.dtypes)

# ----------------------------
# STEP 2: LOAD ONE FULL DAY
# ----------------------------
print("\n===== BUILDING ONE DAY SAMPLE =====")

day_map = {}

for f in files:
    try:
        base = os.path.basename(f)
        date = base.split("-TeamB")[0]  # extracts YYYY-MM-DD

        if date not in day_map:
            day_map[date] = []
        day_map[date].append(f)

    except:
        continue

sample_day = list(day_map.keys())[0]
day_files = day_map[sample_day]

print(f"\nSelected day: {sample_day}")
print(f"Files: {len(day_files)}")

day_frames = []

for f in day_files:
    try:
        df = pd.read_parquet(f)
        df["source_file"] = os.path.basename(f)
        day_frames.append(df)
    except Exception as e:
        print("Skip:", f, e)

if len(day_frames) > 0:
    day_data = pd.concat(day_frames, ignore_index=True)

    print("\n===== ONE DAY DATA SAMPLE =====")
    print(day_data.head(30))

    print("\nSHAPE:", day_data.shape)

    print("\nCOLUMNS:")
    print(day_data.columns)

else:
    print("No data loaded for day sample")

# ----------------------------
# DONE
# ----------------------------
print("\nDONE")