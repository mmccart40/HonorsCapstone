import os
import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ======================
# CONFIG
# ======================
OBJECTIVE_ROOT = "/scratch/user/u.mm342941/objective-TeamA-2020"
SUBJECTIVE_ROOT = "./subjective"
MAX_FILES = 800
FUTURE_DAYS = 7

# ======================
# LOAD SUBJECTIVE
# ======================
def load_subjective():
    dfs = []

    def load_csv(path, name):
        if not os.path.exists(path):
            print(f"Missing: {path}")
            return None
        df = pd.read_csv(path)
        print(f"\nLoading {name}: {df.shape}")
        df["date"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y", errors="coerce")
        df = df.dropna(subset=["date"])
        return df

    injury = load_csv(f"{SUBJECTIVE_ROOT}/injury/injury.csv", "injury")
    performance = load_csv(f"{SUBJECTIVE_ROOT}/game-performance/game-performance.csv", "performance")
    illness = load_csv(f"{SUBJECTIVE_ROOT}/illness/illness.csv", "illness")

    if injury is not None:
        injury["injury_flag"] = 1
        dfs.append(injury[["player_name", "date", "injury_flag"]])

    if performance is not None:
        dfs.append(performance)

    if illness is not None:
        dfs.append(illness)

    subj = pd.concat(dfs, ignore_index=True)
    subj = subj.groupby(["player_name", "date"]).first().reset_index()

    print("\nSubjective rows:", len(subj))
    print("Unique players:", subj["player_name"].nunique())
    print("Date range:", subj["date"].min(), "to", subj["date"].max())

    return subj

# ======================
# LOAD OBJECTIVE (FAST + SAFE)
# ======================
def load_objective(valid_players):
    files = glob.glob(f"{OBJECTIVE_ROOT}/**/*.parquet", recursive=True)
    print("\nTotal parquet files:", len(files))

    rows = []

    for i, f in enumerate(files[:MAX_FILES]):
        if i % 50 == 0:
            print(f"[{i}/{MAX_FILES}] Processing")

        try:
            df = pd.read_parquet(f)

            # handle time column safely
            if "time" in df.columns:
                if np.issubdtype(df["time"].dtype, np.number):
                    df["date"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                else:
                    df["date"] = pd.to_datetime(df["time"], errors="coerce")
            else:
                continue

            df = df.dropna(subset=["date"])

            # extract player_name from file path if missing
            if "player_name" not in df.columns:
                fname = os.path.basename(f)
                df["player_name"] = fname.split("-")[2] if "-" in fname else "unknown"

            df = df[df["player_name"].isin(valid_players)]

            if df.empty:
                continue

            # reduce memory by aggregating immediately
            agg = df.groupby(["player_name", df["date"].dt.date]).agg({
                "speed": ["mean", "max"],
                "heart_rate": ["mean", "max"],
                "accl_x": "mean",
                "accl_y": "mean",
                "accl_z": "mean"
            })

            agg.columns = ["_".join(col) for col in agg.columns]
            agg = agg.reset_index()
            agg.rename(columns={"date": "date"}, inplace=True)
            agg["date"] = pd.to_datetime(agg["date"])

            rows.append(agg)

        except Exception:
            continue

    if len(rows) == 0:
        print("ERROR: No valid objective data found")
        return pd.DataFrame()

    obj = pd.concat(rows, ignore_index=True)
    print("\nObjective rows:", len(obj))
    print("Objective date range:", obj["date"].min(), "to", obj["date"].max())

    return obj

# ======================
# FEATURE ENGINEERING
# ======================
def engineer_features(df):
    df = df.sort_values(["player_name", "date"])

    # rolling workload features
    for col in ["speed_mean", "heart_rate_mean"]:
        df[f"{col}_7d"] = df.groupby("player_name")[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df[f"{col}_28d"] = df.groupby("player_name")[col].transform(
            lambda x: x.rolling(28, min_periods=1).mean()
        )

        df[f"{col}_acwr"] = df[f"{col}_7d"] / (df[f"{col}_28d"] + 1e-5)

    return df

# ======================
# CREATE LABELS (FORWARD LOOKING)
# ======================
def create_labels(df):
    df = df.sort_values(["player_name", "date"])

    df["injury_future"] = df.groupby("player_name")["injury_flag"].transform(
        lambda x: x.rolling(FUTURE_DAYS, min_periods=1).max().shift(-FUTURE_DAYS)
    )

    df["injury_future"] = df["injury_future"].fillna(0)

    return df

# ======================
# IMPUTATION (CRITICAL)
# ======================
def impute_data(df):
    df = df.sort_values(["player_name", "date"])

    # forward fill per player
    df = df.groupby("player_name").apply(lambda g: g.ffill()).reset_index(drop=True)

    # backward fill per player
    df = df.groupby("player_name").apply(lambda g: g.bfill()).reset_index(drop=True)

    # fill remaining with global medians
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df

# ======================
# TRAIN MODEL
# ======================
def train_model(df):
    df = df.dropna(subset=["injury_future"])

    features = [
        "speed_mean", "speed_max",
        "heart_rate_mean", "heart_rate_max",
        "accl_x_mean", "accl_y_mean", "accl_z_mean",
        "speed_mean_7d", "speed_mean_28d", "speed_mean_acwr",
        "heart_rate_mean_7d", "heart_rate_mean_28d", "heart_rate_mean_acwr"
    ]

    X = df[features]
    y = df["injury_future"]

    print("\nPositive rate:", y.mean())

    if y.nunique() < 2:
        print("ERROR: Only one class present. Cannot train model")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Results:")
    print(classification_report(y_test, y_pred))

# ======================
# MAIN
# ======================
def main():
    subj = load_subjective()
    valid_players = subj["player_name"].unique()

    obj = load_objective(valid_players)

    if obj.empty:
        print("Stopping due to no objective data")
        return

    merged = pd.merge(obj, subj, on=["player_name", "date"], how="left")

    merged["injury_flag"] = merged["injury_flag"].fillna(0)

    merged = engineer_features(merged)
    merged = create_labels(merged)
    merged = impute_data(merged)

    print("\nFinal dataset shape:", merged.shape)

    train_model(merged)

if __name__ == "__main__":
    main()