import os
import gc
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
DATA_FOLDER = "objective-TeamB-2020"

# If you only need specific columns, add them here to reduce memory
# Example:
# USE_COLS = ["athlete_id", "date", "load"]
USE_COLS = None


# ----------------------------
# LOAD FILE LIST
# ----------------------------
files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".parquet")]

print(f"Found {len(files)} files")


# ----------------------------
# PROCESS EACH FILE SAFELY
# ----------------------------
def process(df):
    """
    Replace this function with your real pipeline logic.
    Keep it memory-light.
    """

    # Example safe operations (no big copies)
    print("Shape:", df.shape)

    # Example aggregation (replace with your model logic)
    if "athlete_id" in df.columns:
        return df.groupby("athlete_id").size()

    return None


results = []

for i, file in enumerate(files):
    file_path = os.path.join(DATA_FOLDER, file)
    print(f"\n[{i+1}/{len(files)}] Loading {file_path}")

    # ----------------------------
    # MEMORY-SAFE LOADING
    # ----------------------------
    if USE_COLS:
        df = pd.read_parquet(file_path, columns=USE_COLS)
    else:
        df = pd.read_parquet(file_path)

    # ----------------------------
    # PROCESS IMMEDIATELY (NO STORAGE OF FULL DATAFRAMES)
    # ----------------------------
    result = process(df)

    if result is not None:
        results.append(result)

    # ----------------------------
    # HARD MEMORY CLEANUP
    # ----------------------------
    del df
    gc.collect()


# ----------------------------
# FINAL COMBINATION (SAFE)
# ----------------------------
print("\nCombining results...")

if results:
    final_result = pd.concat(results)
    print(final_result.head())
else:
    print("No results generated")


print("DONE")