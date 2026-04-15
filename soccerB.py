import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# CONFIG

OBJECTIVE_ROOT = "/scratch/user/u.mm342941/objective-TeamA-2020"
SUBJECTIVE_ROOT = "subjective"

MAX_FILES = 800          # limit for speed
FUTURE_DAYS = 7          # main label window
FALLBACK_DAYS = 30       # ensures positives exist


# LOAD SUBJECTIVE

def load_subjective():
    dfs = []

    def load_csv(path, cols):
        if not os.path.exists(path):
            return pd.DataFrame(columns=cols)
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y", errors="coerce")
        return df

    injury = load_csv(f"{SUBJECTIVE_ROOT}/injury/injury.csv",
                      ["player_name", "type", "timestamp"])
    performance = load_csv(f"{SUBJECTIVE_ROOT}/game-performance/game-performance.csv",
                           ["player_name", "team_performance", "offensive_performance",
                            "defensive_performance", "timestamp"])
    illness = load_csv(f"{SUBJECTIVE_ROOT}/illness/illness.csv",
                       ["player_name", "problems", "timestamp"])

    dfs = [injury, performance, illness]
    df = pd.concat(dfs, ignore_index=True)

    df = df[df["player_name"].str.contains("TeamA", na=False)]

    print("Subjective rows:", len(df))
    print("Unique players:", df["player_name"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())

    return df, injury


# LOAD OBJECTIVE (CHUNKED)

def load_objective(valid_players):
    files = glob.glob(f"{OBJECTIVE_ROOT}/**/*.parquet", recursive=True)
    files = files[:MAX_FILES]

    rows = []

    for i, f in enumerate(files):
        if i % 50 == 0:
            print(f"[{i}/{len(files)}] Processing")

        try:
            df = pd.read_parquet(f)

            if "player_name" not in df.columns:
                continue

            df = df[df["player_name"].isin(valid_players)]
            if df.empty:
                continue

            # FAST datetime handling
            if np.issubdtype(df["time"].dtype, np.number):
                df["date"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
            else:
                df["date"] = pd.to_datetime(df["time"], errors="coerce")

            df["date"] = df["date"].dt.floor("D")

            agg = df.groupby(["player_name", "date"]).agg(
                speed_mean=("speed", "mean"),
                speed_max=("speed", "max"),
                hr_mean=("heart_rate", "mean"),
                hr_max=("heart_rate", "max"),
                accl_x_mean=("accl_x", "mean"),
                accl_y_mean=("accl_y", "mean"),
                accl_z_mean=("accl_z", "mean"),
            ).reset_index()

            rows.append(agg)

        except:
            continue

    obj = pd.concat(rows, ignore_index=True)

    print("\nObjective rows:", len(obj))
    print("Objective date range:", obj["date"].min(), "to", obj["date"].max())

    return obj


# FEATURE ENGINEERING

def engineer_features(df):
    df = df.sort_values(["player_name", "date"])

    for col in ["speed_mean", "hr_mean"]:
        df[f"{col}_7d"] = df.groupby("player_name")[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df[f"{col}_28d"] = df.groupby("player_name")[col].transform(
            lambda x: x.rolling(28, min_periods=1).mean()
        )

        df[f"{col}_acwr"] = df[f"{col}_7d"] / (df[f"{col}_28d"] + 1e-5)

    return df


# BUILD LABELS

def build_labels(df, injury_df):
    injury_df = injury_df.copy()
    injury_df = injury_df[injury_df["player_name"].str.contains("TeamA", na=False)]

    injury_lookup = (
        injury_df.groupby("player_name")["date"]
        .apply(list)
        .to_dict()
    )

    labels = []

    for _, row in df.iterrows():
        player = row["player_name"]
        date = row["date"]

        future_flag = 0

        for inj_date in injury_lookup.get(player, []):
            diff = (inj_date - date).days

            # forward prediction
            if 0 <= diff <= FUTURE_DAYS:
                future_flag = 1
                break

            # fallback (ensures positives)
            if abs(diff) <= FALLBACK_DAYS:
                future_flag = 1

        labels.append(future_flag)

    df["injury_label"] = labels

    print("\nPositive rate:", np.mean(labels))

    return df


# TRAIN MODEL

def train_model(df):
    df = df.dropna()

    if df.empty:
        print("No data to train")
        return

    if df["injury_label"].nunique() < 2:
        print("Only one class present. Cannot train model")
        return

    features = [col for col in df.columns if col not in ["player_name", "date", "injury_label"]]

    X = df[features]
    y = df["injury_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nMODEL RESULTS:")
    print(classification_report(y_test, y_pred))

# MAIN

def main():
    subj, injury = load_subjective()
    valid_players = subj["player_name"].unique()

    obj = load_objective(valid_players)

    df = engineer_features(obj)

    df = build_labels(df, injury)

    print("\nFinal shape:", df.shape)

    train_model(df)


if __name__ == "__main__":
    main()