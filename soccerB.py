import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------
OBJECTIVE_PATH = "/scratch/user/u.mm342941/objective-TeamA-2020"
SUBJECTIVE_PATH = "subjective"
MAX_FILES = 800
CHUNK_SIZE = 50

# -----------------------------
# LOAD SUBJECTIVE DATA
# -----------------------------
def load_subjective():
    dfs = []

    def load_csv(path, name):
        if not os.path.exists(path):
            print(f"Missing: {path}")
            return None

        df = pd.read_csv(path)
        print(f"\nLoading subjective: {name}")
        print("Raw shape:", df.shape)

        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(
                df["timestamp"],
                format="%d.%m.%Y",
                errors="coerce"
            )
        else:
            return None

        df = df.dropna(subset=["player_name", "date"])
        print(f"Rows after cleaning: {len(df)}")
        return df

    dfs.append(load_csv(f"{SUBJECTIVE_PATH}/injury/injury.csv", "injury"))
    dfs.append(load_csv(f"{SUBJECTIVE_PATH}/game-performance/game-performance.csv", "performance"))
    dfs.append(load_csv(f"{SUBJECTIVE_PATH}/illness/illness.csv", "illness"))

    dfs = [d for d in dfs if d is not None]
    df = pd.concat(dfs, ignore_index=True)

    print("\nSubjective rows:", len(df))
    print("Unique players:", df["player_name"].nunique())

    # filter TeamA only
    df = df[df["player_name"].str.startswith("TeamA")]

    print("Filtered TeamA players:", df["player_name"].nunique())

    return df


# -----------------------------
# BUILD INJURY LOOKUP
# -----------------------------
def build_injury_lookup(df):
    injury_df = df[df["type"].notna()].copy()
    lookup = {}

    for player, group in injury_df.groupby("player_name"):
        lookup[player] = group["date"].tolist()

    return lookup


# -----------------------------
# PROCESS OBJECTIVE DATA (CHUNKED)
# -----------------------------
def process_objective():
    files = glob.glob(f"{OBJECTIVE_PATH}/**/*.parquet", recursive=True)
    print("\nTotal parquet files:", len(files))

    files = files[:MAX_FILES]

    results = []

    for i in range(0, len(files), CHUNK_SIZE):
        chunk_files = files[i:i + CHUNK_SIZE]
        chunk_df_list = []

        print(f"[{min(i+CHUNK_SIZE, len(files))}/{len(files)}] Processing")

        for f in chunk_files:
            try:
                df = pd.read_parquet(f)

                if "player_name" not in df.columns:
                    continue

                df = df[df["player_name"].str.startswith("TeamA")]

                if len(df) == 0:
                    continue

                # FAST DATE PARSING
                if "time" in df.columns:
                    df["date"] = pd.to_datetime(df["time"], errors="coerce")
                else:
                    continue

                df = df.dropna(subset=["date"])
                df["date"] = df["date"].dt.floor("D")

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

                chunk_df_list.append(agg)

            except Exception:
                continue

        if chunk_df_list:
            results.append(pd.concat(chunk_df_list, ignore_index=True))

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    print("\nObjective rows:", len(df))
    return df


# -----------------------------
# LABEL DATA (FIXED)
# -----------------------------
def label_data(objective_df, injury_lookup):
    objective_df["injury_label"] = 0

    for idx, row in objective_df.iterrows():
        player = row["player_name"]
        date = row["date"]

        if player not in injury_lookup:
            continue

        for inj_date in injury_lookup[player]:
            if abs((inj_date - date).days) <= 30:
                objective_df.at[idx, "injury_label"] = 1
                break

    return objective_df


# -----------------------------
# MAIN
# -----------------------------
def main():
    subjective_df = load_subjective()
    injury_lookup = build_injury_lookup(subjective_df)

    objective_df = process_objective()

    if len(objective_df) == 0:
        print("No objective data")
        return

    objective_df = label_data(objective_df, injury_lookup)

    print("\nPositive rate:", objective_df["injury_label"].mean())

    # -----------------------------
    # MODEL
    # -----------------------------
    features = [
        "speed_mean", "speed_max",
        "hr_mean", "hr_max",
        "accl_x_mean", "accl_y_mean", "accl_z_mean"
    ]

    X = objective_df[features].fillna(0)
    y = objective_df["injury_label"]

    if y.nunique() < 2:
        print("ERROR: Only one class present. Cannot train model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    try:
        print("ROC AUC:", roc_auc_score(y_test, y_prob))
    except:
        print("ROC AUC could not be computed")


if __name__ == "__main__":
    main()