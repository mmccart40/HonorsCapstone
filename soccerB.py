import pandas as pd
import glob
import os

# -----------------------------
# PATHS
# -----------------------------
OBJECTIVE_PATH = "/scratch/user/u.mm342941/objective-TeamA-2020"
SUBJECTIVE_PATH = "subjective"

# -----------------------------
# LOAD SUBJECTIVE DATA
# -----------------------------
def load_subjective():
    dfs = []

    files = [
        ("injury", "injury/injury.csv"),
        ("performance", "game-performance/game-performance.csv"),
        ("illness", "illness/illness.csv")
    ]

    for name, path in files:
        full_path = os.path.join(SUBJECTIVE_PATH, path)

        if not os.path.exists(full_path):
            print("Missing:", full_path)
            continue

        df = pd.read_csv(full_path)

        # Parse European date format
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(
                df["timestamp"],
                format="%d.%m.%Y",
                errors="coerce"
            )

        df = df.dropna(subset=["date"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    print("\nSubjective rows:", len(df))
    print("Unique players:", df["player_name"].nunique())

    return df

subjective_df = load_subjective()

# -----------------------------
# FILTER TO TEAMA PLAYERS
# -----------------------------
subjective_df = subjective_df[
    subjective_df["player_name"].str.contains("TeamA")
]

teamA_players = set(subjective_df["player_name"])

print("Filtered TeamA players:", len(teamA_players))

# -----------------------------
# LOAD OBJECTIVE DATA (SMART)
# -----------------------------
files = glob.glob(OBJECTIVE_PATH + "/**/*.parquet", recursive=True)
print("Total parquet files:", len(files))

# LIMIT FOR SPEED (increase later)
files = files[:300]

results = []

for i, f in enumerate(files):
    if (i + 1) % 20 == 0:
        print(f"[{i+1}/{len(files)}] Processing")

    try:
        df = pd.read_parquet(f)

        if "player_name" not in df.columns:
            continue

        # FILTER EARLY
        df = df[df["player_name"].isin(teamA_players)]
        if df.empty:
            continue

        # HANDLE DATE
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
        elif "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            continue

        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.floor("D")

        # AGGREGATE IMMEDIATELY
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
            "hr_mean", "hr_max",
            "accl_x_mean", "accl_y_mean", "accl_z_mean"
        ]

        results.append(agg)

    except Exception:
        continue

# Combine objective
objective_df = pd.concat(results, ignore_index=True)

print("\nObjective rows:", len(objective_df))

# -----------------------------
# ALIGN DATE TYPES
# -----------------------------
objective_df["date"] = pd.to_datetime(objective_df["date"])
subjective_df["date"] = pd.to_datetime(subjective_df["date"])

# -----------------------------
# MERGE (INNER JOIN)
# -----------------------------
merged = pd.merge(
    objective_df,
    subjective_df,
    on=["player_name", "date"],
    how="inner"
)

# -----------------------------
# RESULTS
# -----------------------------
print("\nFinal shape:", merged.shape)

print("\nMerge coverage:")
print(merged.notnull().mean())

print("\nSample:")
print(merged.head())

# Save
merged.to_csv("final_dataset.csv", index=False)

print("\nDone")