import os
import pandas as pd

# -------------------------
# ROOT PATH (ACES)
# -------------------------
root_path = root_path = "/scratch/user/u.mm342941/objective-TeamB-2020/2020"

print("Root path:", root_path)

dfs = []
bad_files = []

# -------------------------
# RECURSIVE WALK THROUGH ALL FOLDERS
# -------------------------
for dirpath, dirnames, filenames in os.walk(root_path):

    for file in filenames:

        if not file.endswith(".parquet"):
            continue

        file_path = os.path.join(dirpath, file)
        print("Loading:", file_path)

        try:
            df_temp = pd.read_parquet(file_path)

            # -------------------------
            # Extract date from folder path
            # Example: .../2020-06-01/
            # -------------------------
            date_str = os.path.basename(dirpath)
            df_temp["date"] = pd.to_datetime(date_str, errors="coerce")

            dfs.append(df_temp)

        except Exception as e:
            print("Failed:", file_path)
            print("Reason:", e)
            bad_files.append(file_path)

# -------------------------
# COMBINE DATA
# -------------------------
if dfs:
    df = pd.concat(dfs, ignore_index=True)

    print("\nSUCCESS")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

else:
    raise ValueError("No valid parquet files loaded")

# -------------------------
# BAD FILE REPORT
# -------------------------
print("\nBad files:", len(bad_files))
for f in bad_files:
    print(f)

# -------------------------
# SAVE FOR FAST RELOAD
# -------------------------
output_path = os.path.join(root_path, "TeamB_2020_combined.parquet")
df.to_parquet(output_path)

print("\nSaved combined dataset to:", output_path)
