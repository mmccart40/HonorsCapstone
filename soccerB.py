import pandas as pd
import glob
import os
import gc

# -----------------------
# CONFIG
# -----------------------
OBJECTIVE_PATH = "/scratch/user/u.mm342941/objective-TeamA-2020/**/*.parquet"
CHUNK_SIZE = 50
OUTPUT_FILE = "objective_aggregated.csv"

# -----------------------
# LOAD SUBJECTIVE
# -----------------------
def load_subjective():
    dfs = []

    def load_csv(path):
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y", errors="coerce")
            return df
        return pd.DataFrame()

    injury = load_csv("subjective/injury/injury.csv")
    performance = load_csv("subjective/game-performance/performance.csv")
    illness = load_csv("subjective/illness/illness.csv")

    dfs = [df for df in [injury, performance, illness] if not df.empty]
    subjective = pd.concat(dfs, ignore_index=True)

    return subjective

subjective = load_subjective()
print("Subjective rows:", len(subjective))

# -----------------------
# PROCESS OBJECTIVE IN CHUNKS
# -----------------------
files = glob.glob(OBJECTIVE_PATH, recursive=True)
print("Total files:", len(files))

# remove old file
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

def process_chunk(file_chunk):
    results = []

    for f in file_chunk:
        try:
            df = pd.read_parquet(f)

            if "player_name" not in df.columns:
                continue

            # DATE HANDLING
            if "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
            elif "time" in df.columns:
                df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.date
            else:
                continue

            df = df.dropna(subset=["date", "player_name"])

            # AGGREGATE
            agg = df.groupby(["player_name", "date"]).agg({
                "speed": ["mean", "max"],
                "heart_rate": ["mean", "max"],
                "accl_x": "mean",
                "accl_y": "mean",
                "accl_z": "mean"
            }).reset_index()

            agg.columns = [
                "player_name", "date",
                "speed_mean", "speed_max",
                "heart_rate_mean", "heart_rate_max",
                "accl_x_mean", "accl_y_mean", "accl_z_mean"
            ]

            results.append(agg)

        except Exception:
            continue

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

# -----------------------
# RUN CHUNKS
# -----------------------
for i in range(0, len(files), CHUNK_SIZE):
    chunk = files[i:i+CHUNK_SIZE]
    print(f"Processing chunk {i//CHUNK_SIZE + 1}")

    df_chunk = process_chunk(chunk)

    if not df_chunk.empty:
        df_chunk.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not os.path.exists(OUTPUT_FILE),
            index=False
        )

    del df_chunk
    gc.collect()

# -----------------------
# LOAD AGGREGATED OBJECTIVE
# -----------------------
objective = pd.read_csv(OUTPUT_FILE)
objective["date"] = pd.to_datetime(objective["date"])

# -----------------------
# CLEAN PLAYER NAMES
# -----------------------
def clean_name(x):
    return str(x).replace("TeamA-", "").replace("TeamB-", "")

objective["player_id"] = objective["player_name"].apply(clean_name)
subjective["player_id"] = subjective["player_name"].apply(clean_name)

# -----------------------
# MERGE (IMPORTANT FIX)
# -----------------------
merged = pd.merge(
    objective,
    subjective,
    on=["player_id", "date"],
    how="left"
)

# -----------------------
# CREATE LABEL
# -----------------------
merged["injury_flag"] = merged["type"].notna().astype(int)

print("\nFinal shape:", merged.shape)
print("\nInjury rate:", merged["injury_flag"].mean())

# -----------------------
# SAVE FINAL DATASET
# -----------------------
merged.to_csv("final_dataset.csv", index=False)

print("Done")