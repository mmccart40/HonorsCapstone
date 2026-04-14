import pandas as pd
import numpy as np
import glob
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

OBJECTIVE_PATH = "/scratch/user/u.mm342941/objective-TeamA-2020/**/*.parquet"

SUBJECTIVE_PATHS = {
    "injury": "subjective/injury/injury.csv",
    "performance": "subjective/game-performance/game-performance.csv",
    "illness": "subjective/illness/illness.csv"
}

# -------------------------
# CLEAN PLAYER IDS
# -------------------------
def clean_player_name(name):
    if pd.isna(name):
        return name
    name = str(name)
    name = name.replace("TeamA-TeamA-", "TeamA-")
    return name

# -------------------------
# LOAD SUBJECTIVE DATA
# -------------------------
def load_subjective():
    dfs = []

    for key, path in SUBJECTIVE_PATHS.items():
        if not os.path.exists(path):
            print("Missing:", path)
            continue

        print("\nLoading subjective:", key)
        df = pd.read_csv(path)

        df["player_name"] = df["player_name"].apply(clean_player_name)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            format="%d.%m.%Y",
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].dt.floor("D")

        dfs.append(df)

        print("Rows:", len(df))

    full = pd.concat(dfs, ignore_index=True)

    print("\nSubjective rows:", len(full))
    print("Unique players:", full["player_name"].nunique())

    return full

# -------------------------
# LOAD OBJECTIVE DATA
# -------------------------
def load_objective():
    files = sorted(glob.glob(OBJECTIVE_PATH, recursive=True))

    print("\nTotal objective files:", len(files))

    dfs = []

    for i, f in enumerate(files):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(files)}")

        try:
            df = pd.read_parquet(f)

            if "time" not in df.columns:
                continue

            df["player_name"] = df["player_name"].apply(clean_player_name)

            df["date"] = pd.to_datetime(df["time"], unit="s").dt.floor("D")

            dfs.append(df)

        except:
            continue

    full = pd.concat(dfs, ignore_index=True)

    print("\nObjective shape:", full.shape)
    print("Date range:", full["date"].min(), "to", full["date"].max())

    return full

# -------------------------
# FEATURE ENGINEERING
# -------------------------
def build_features(df):
    df = df.sort_values(["player_name", "date"])

    # rolling features per player
    df["speed_mean_3d"] = df.groupby("player_name")["speed"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    df["speed_std_3d"] = df.groupby("player_name")["speed"].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    )

    df["hr_mean_3d"] = df.groupby("player_name")["heart_rate"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    df["accl_mag"] = np.sqrt(df["accl_x"]**2 + df["accl_y"]**2 + df["accl_z"]**2)

    df["accl_mean_3d"] = df.groupby("player_name")["accl_mag"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    return df

# -------------------------
# CREATE INJURY LABEL
# -------------------------
def create_labels(objective, subjective):
    injury = subjective[subjective["type"].notna()].copy()

    injury_dates = injury.groupby("player_name")["date"].apply(list).to_dict()

    labels = []

    for _, row in objective.iterrows():
        player = row["player_name"]
        date = row["date"]

        future_injury = False

        if player in injury_dates:
            for d in injury_dates[player]:
                if date < d <= date + pd.Timedelta(days=7):
                    future_injury = True
                    break

        labels.append(int(future_injury))

    objective["injury_next_7d"] = labels

    return objective

# -------------------------
# BUILD MODEL DATASET
# -------------------------
def build_dataset(obj, sub):
    obj = build_features(obj)
    obj = create_labels(obj, sub)

    features = [
        "speed",
        "heart_rate",
        "accl_x",
        "accl_y",
        "accl_z",
        "speed_mean_3d",
        "speed_std_3d",
        "hr_mean_3d",
        "accl_mean_3d"
    ]

    obj = obj.dropna(subset=features + ["injury_next_7d"])

    X = obj[features]
    y = obj["injury_next_7d"]

    return X, y

# -------------------------
# TRAIN MODEL
# -------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nMODEL RESULTS")
    print(classification_report(y_test, preds))

    return model

# -------------------------
# MAIN
# -------------------------
def main():
    print("Loading subjective...")
    subjective = load_subjective()

    print("\nLoading objective...")
    objective = load_objective()

    print("\nBuilding dataset...")
    X, y = build_dataset(objective, subjective)

    print("\nFinal dataset size:", X.shape)

    print("\nTraining model...")
    model = train_model(X, y)

    print("\nDone")

if __name__ == "__main__":
    main()