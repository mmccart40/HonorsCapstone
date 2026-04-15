import os
import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUBJECTIVE_ROOT = os.path.join(BASE_DIR, "subjective")
OBJECTIVE_ROOT = "/scratch/user/u.mm342941/objective-TeamA-2020"

START_DATE = "2020-06-01"
END_DATE = "2020-12-31"

FUTURE_DAYS = 7
MAX_FILES = 1000


# ---------------------------
# SUBJECTIVE DATA
# ---------------------------
def load_subjective():
    print("\nLoading subjective data")

    injury_path = os.path.join(SUBJECTIVE_ROOT, "injury", "injury.csv")
    illness_path = os.path.join(SUBJECTIVE_ROOT, "illness", "illness.csv")
    perf_path = os.path.join(SUBJECTIVE_ROOT, "game-performance", "performance.csv")

    injury = pd.read_csv(injury_path)
    illness = pd.read_csv(illness_path)
    perf = pd.read_csv(perf_path)

    def parse(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        return df.dropna(subset=["timestamp"])

    injury = parse(injury)
    illness = parse(illness)
    perf = parse(perf)

    injury["event"] = 1
    illness["event"] = 0
    perf["event"] = 0

    df = pd.concat([
        injury[["player_name", "timestamp", "event"]],
        illness[["player_name", "timestamp", "event"]],
        perf[["player_name", "timestamp", "event"]],
    ])

    # filter TeamA only (optional adjust if needed)
    df = df[df["player_name"].str.contains("TeamA", na=False)]

    # IMPORTANT FILTER WINDOW
    df = df[
        (df["timestamp"] >= START_DATE) &
        (df["timestamp"] <= END_DATE)
    ]

    print("Subjective rows:", len(df))
    print("Injury events:", df["event"].sum())
    print("Players:", df["player_name"].nunique())
    print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())

    return df[df["event"] == 1]


# ---------------------------
# OBJECTIVE DATA
# ---------------------------
def extract_date_from_path(path):
    parts = path.split(os.sep)
    for p in parts:
        try:
            return pd.to_datetime(p).date()
        except:
            continue
    return None


def load_objective():
    print("\nLoading objective data")

    files = glob.glob(os.path.join(OBJECTIVE_ROOT, "**/*.parquet"), recursive=True)
    print("Total parquet files:", len(files))

    dfs = []

    for i, f in enumerate(files[:MAX_FILES]):
        if i % 200 == 0:
            print(f"[{i}/{len(files)}] Processing")

        try:
            df = pd.read_parquet(f)

            if "time" not in df.columns:
                continue

            file_date = extract_date_from_path(f)
            if file_date is None:
                continue

            df["date"] = file_date

            # FIX: combine folder date + time column
            df["timestamp"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce"
            )

            df = df.dropna(subset=["timestamp"])

            dfs.append(df)

        except Exception:
            continue

    if len(dfs) == 0:
        raise ValueError("No objective data loaded. Check parquet structure or path parsing.")

    obj = pd.concat(dfs, ignore_index=True)

    print("\nObjective summary")
    print("Rows:", len(obj))
    print("Players:", obj["player_name"].nunique())
    print("Date range:", obj["timestamp"].min(), "to", obj["timestamp"].max())

    return obj


# ---------------------------
# AGGREGATION
# ---------------------------
def aggregate_daily(obj):
    print("\nAggregating daily workload")

    obj["date"] = obj["timestamp"].dt.date

    features = [c for c in ["speed", "heart_rate", "accl_x", "accl_y", "accl_z"] if c in obj.columns]

    daily = obj.groupby(["player_name", "date"])[features].mean().reset_index()

    print("Daily rows:", len(daily))
    return daily


# ---------------------------
# LABELING
# ---------------------------
def build_labels(daily, injuries):
    print("\nBuilding labels")

    injuries["date"] = injuries["timestamp"].dt.date

    injury_map = injuries.groupby("player_name")["date"].apply(list).to_dict()

    labels = []

    for _, row in daily.iterrows():
        pid = row["player_name"]
        d = row["date"]

        label = 0

        if pid in injury_map:
            for inj in injury_map[pid]:
                delta = (inj - d).days
                if 0 <= delta <= FUTURE_DAYS:
                    label = 1
                    break

        labels.append(label)

    daily["label"] = labels

    print("Positive samples:", daily["label"].sum())
    print("Positive rate:", round(daily["label"].mean(), 4))

    return daily


# ---------------------------
# IMPUTATION
# ---------------------------
def impute(df):
    print("\nImputing missing values")

    for c in df.columns:
        if c in ["player_name", "date", "label"]:
            continue

        df[c] = df.groupby("player_name")[c].transform(lambda x: x.fillna(x.mean()))

    df = df.fillna(df.median(numeric_only=True))
    return df


# ---------------------------
# MODEL
# ---------------------------
def run_model(df):
    print("\nTraining model")

    feature_cols = [c for c in df.columns if c not in ["player_name", "date", "label"]]

    X = df[feature_cols]
    y = df["label"]

    if y.nunique() < 2:
        print("Only one class present")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds, zero_division=0))


# ---------------------------
# SAMPLE OUTPUT
# ---------------------------
def show_sample(df):
    print("\nSample athlete timeline")

    sample_player = df["player_name"].iloc[0]

    sample = df[df["player_name"] == sample_player].sort_values("date")

    print("\nPlayer:", sample_player)
    print(sample.head(20))


# ---------------------------
# MAIN
# ---------------------------
def main():
    injuries = load_subjective()
    obj = load_objective()

    daily = aggregate_daily(obj)

    injuries = injuries[
        (injuries["timestamp"] >= START_DATE) &
        (injuries["timestamp"] <= END_DATE)
    ]

    print("\nFiltered injuries:", len(injuries))

    daily = build_labels(daily, injuries)
    daily = impute(daily)

    print("\nFinal dataset shape:", daily.shape)

    show_sample(daily)

    run_model(daily)


if __name__ == "__main__":
    main()