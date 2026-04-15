import pandas as pd
import numpy as np
import glob
import os

OBJECTIVE_PATH = "/scratch/user/u.mm342941/objective-TeamA-2020/**/*.parquet"

SUBJECTIVE_PATHS = {
    "injury": "subjective/injury/injury.csv",
    "performance": "subjective/game-performance/game-performance.csv",
    "illness": "subjective/illness/illness.csv"
}

WINDOW_DAYS = 7

def clean_player_name(name):
    if pd.isna(name):
        return name
    return str(name).replace("TeamA-TeamA-", "TeamA-")

# ----------------------------
# SUBJECTIVE
# ----------------------------
def load_subjective():
    dfs = []

    for key, path in SUBJECTIVE_PATHS.items():
        if not os.path.exists(path):
            print("Missing:", path)
            continue

        print("\nLoading:", key)
        df = pd.read_csv(path)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            format="%d.%m.%Y",
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].dt.floor("D")
        df["player_name"] = df["player_name"].apply(clean_player_name)

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    print("Subjective rows:", len(combined))
    return combined

# ----------------------------
# OBJECTIVE (MEMORY SAFE)
# ----------------------------
def process_objective():
    files = glob.glob(OBJECTIVE_PATH, recursive=True)

    print("Total files:", len(files))

    aggregated_rows = []

    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}]")

        try:
            df = pd.read_parquet(f)

            if "time" not in df.columns:
                continue

            df["date"] = pd.to_datetime(df["time"], unit="s").dt.floor("D")
            df["player_name"] = df["player_name"].apply(clean_player_name)

            # compute features
            df["acc_mag"] = np.sqrt(
                df["accl_x"]**2 + df["accl_y"]**2 + df["accl_z"]**2
            )

            agg = df.groupby(["player_name", "date"]).agg({
                "speed": ["mean", "max"],
                "heart_rate": ["mean", "max"],
                "acc_mag": "mean"
            }).reset_index()

            agg.columns = [
                "player_name", "date",
                "speed_mean", "speed_max",
                "heart_rate_mean", "heart_rate_max",
                "acc_mag_mean"
            ]

            aggregated_rows.append(agg)

            # IMPORTANT: delete raw df immediately
            del df

        except Exception as e:
            print("Skipping:", f)
            print(e)

    combined = pd.concat(aggregated_rows, ignore_index=True)

    print("Objective rows:", len(combined))
    return combined

# ----------------------------
# LABELING
# ----------------------------
def create_labels(objective, subjective):
    injury = subjective[subjective["type"].notna()]

    objective = objective.sort_values(["player_name", "date"])
    injury = injury.sort_values(["player_name", "date"])

    labels = []

    for _, row in objective.iterrows():
        player = row["player_name"]
        date = row["date"]

        future = injury[
            (injury["player_name"] == player) &
            (injury["date"] > date) &
            (injury["date"] <= date + pd.Timedelta(days=WINDOW_DAYS))
        ]

        labels.append(1 if len(future) > 0 else 0)

    objective["injury_next_7d"] = labels
    return objective

# ----------------------------
# MAIN
# ----------------------------
def main():
    print("Loading subjective...")
    subjective = load_subjective()

    print("\nProcessing objective...")
    objective = process_objective()

    print("\nCreating labels...")
    dataset = create_labels(objective, subjective)

    dataset = dataset.fillna(0)

    print("\nFinal shape:", dataset.shape)

    dataset.to_csv("soccer_mon_ml_ready.csv", index=False)

    print("Saved dataset")

if __name__ == "__main__":
    main()