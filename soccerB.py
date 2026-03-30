import os
import gc
import pandas as pd

# ----------------------------
# FIXED DATA PATH (IMPORTANT)
# ----------------------------
DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"

# Optional: reduce memory if needed
USE_COLS = None  # or ["athlete_id", "date"] if you know needed columns


# ----------------------------
# VALIDATE PATH EARLY (IMPORTANT DEBUG STEP)
# ----------------------------
if not os.path.exists(DATA_FOLDER):
    raise FileNotFoundError(f"DATA_FOLDER not found: {DATA_FOLDER}")


# ----------------------------
# LOAD FILE LIST
# ----------------------------
files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".parquet")]

print(f"Found {len(files)} parquet files")


# ----------------------------
# PROCESS FUNCTION (SAFE PLACEHOLDER)
# ----------------------------
def process(df):
    """
    Replace this with your actual model / feature engineering logic.
    Must stay memory-light.
    """

    print("Processing shape:", df.shape)

    # Example lightweight operation
    if "athlete_id" in df.columns:
        return df.groupby("athlete_id").size()

    return None


results = []


# ----------------------------
# MAIN LOOP (OOM SAFE)
# ----------------------------
for i, file in enumerate(files):
    file_path = os.path.join(DATA_FOLDER, file)
    print(f"\n[{i+1}/{len(files)}] Loading {file_path}")

    # Load parquet safely
    if USE_COLS:
        df = pd.read_parquet(file_path, columns=USE_COLS)
    else:
        df = pd.read_parquet(file_path)

    # Process immediately
    result = process(df)

    if result is not None:
        results.append(result)

    # IMPORTANT: free memory
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