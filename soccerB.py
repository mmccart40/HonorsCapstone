import os
import pandas as pd

# =========================================================
# ACES SAFE CONFIG
# =========================================================

ROOT_PATH = "/scratch/user/u.mm342941/objective-TeamB-2020/2020"

print("Root path:", ROOT_PATH)
print("Exists:", os.path.exists(ROOT_PATH))

if not os.path.exists(ROOT_PATH):
    raise FileNotFoundError(f"Dataset path not found: {ROOT_PATH}")

# =========================================================
# STORAGE (SMALL FOOTPRINT)
# =========================================================

processed_files = 0
failed_files = 0

output_path = os.path.join(ROOT_PATH, "TeamB_2020_combined.parquet")

# Remove old output if exists (avoids appending issues)
if os.path.exists(output_path):
    os.remove(output_path)

first_write = True

# =========================================================
# STREAMING PARQUET PROCESSING (NO OOM)
# =========================================================

for dirpath, dirnames, filenames in os.walk(ROOT_PATH):

    for file in filenames:

        if not file.endswith(".parquet"):
            continue

        file_path = os.path.join(dirpath, file)
        print("Loading:", file_path)

        try:
            # -------------------------------------------------
            # Read parquet (optionally reduce memory here)
            # -------------------------------------------------
            df = pd.read_parquet(file_path)

            # -------------------------------------------------
            # Extract date from folder name
            # -------------------------------------------------
            date_str = os.path.basename(dirpath)
            df["date"] = pd.to_datetime(date_str, errors="coerce")

            # -------------------------------------------------
            # STREAM WRITE (prevents RAM explosion)
            # -------------------------------------------------
            df.to_parquet(
                output_path,
                engine="pyarrow",
                index=False,
                append=not first_write
            )
            first_write = False

            processed_files += 1

            # Free memory immediately
            del df

        except Exception as e:
            print("Failed:", file_path)
            print("Reason:", e)
            failed_files += 1

# =========================================================
# SUMMARY
# =========================================================

print("\n====================")
print("DONE")
print("Processed files:", processed_files)
print("Failed files:", failed_files)
print("Output saved to:", output_path)
print("====================")