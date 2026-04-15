
import os
import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Paths resolved relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECTIVE_ROOT = os.path.join(BASE_DIR, "subjective")
OBJECTIVE_ROOT = "/scratch/user/u.mm342941/objective-TeamA-2020"

FUTURE_DAYS = 7
MAX_PARQUET_FILES = 800

print("BASE_DIR:", BASE_DIR)
print("SUBJECTIVE_ROOT:", SUBJECTIVE_ROOT)
print("SUBJECTIVE_ROOT exists:", os.path.exists(SUBJECTIVE_ROOT))

def load_subjective():
    injury_path = os.path.join(SUBJECTIVE_ROOT, "injury", "injury.csv")
    illness_path = os.path.join(SUBJECTIVE_ROOT, "illness", "illness.csv")
    perf_path = os.path.join(SUBJECTIVE_ROOT, "game-performance", "performance.csv")

    print("Loading subjective files")
    print(injury_path)
    print(illness_path)
    print(perf_path)

    injury = pd.read_csv(injury_path)
    illness = pd.read_csv(illness_path)
    perf = pd.read_csv(perf_path)

    injury["timestamp"] = pd.to_datetime(injury["timestamp"], format="%d.%m.%Y", errors="coerce")
    illness["timestamp"] = pd.to_datetime(illness["timestamp"], format="%d.%m.%Y", errors="coerce")
    perf["timestamp"] = pd.to_datetime(perf["timestamp"], format="%d.%m.%Y", errors="coerce")

    injury["injury"] = 1
    illness["injury"] = 0
    perf["injury"] = 0

    df = pd.concat([
        injury[["player_name", "timestamp", "injury"]],
        illness[["player_name", "timestamp", "injury"]],
        perf[["player_name", "timestamp", "injury"]],
    ], ignore_index=True)

    df = df.dropna(subset=["timestamp"])
    df = df[df["player_name"].str.contains("TeamA", na=False)]

    print("Subjective rows after TeamA filter:", len(df))
    print("Unique subjective players:", df["player_name"].nunique())
    print("Subjective date range:", df["timestamp"].min(), "to", df["timestamp"].max())
    print("Total injury events:", df["injury"].sum())

    return df[df["injury"] == 1]

def load_objective():
    files = glob.glob(os.path.join(OBJECTIVE_ROOT, "**/*.parquet"), recursive=True)
    print("Parquet files found:", len(files))

    rows = []

    for f in files[:MAX_PARQUET_FILES]:
        try:
            df = pd.read_parquet(f)

            if "timestamp" not in df.columns or "player_name" not in df.columns:
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["date"] = df["timestamp"].dt.date
            rows.append(df)

        except Exception:
            continue

    if len(rows) == 0:
        raise ValueError("No valid objective parquet files loaded")

    obj = pd.concat(rows, ignore_index=True)

    print("Objective rows loaded:", len(obj))
    print("Objective players:", obj["player_name"].nunique())
    print("Objective date range:", obj["date"].min(), "to", obj["date"].max())

    return obj

def aggregate_daily(obj):
    num_cols = ["speed", "heart_rate", "accl_x", "accl_y", "accl_z"]
    num_cols = [c for c in num_cols if c in obj.columns]

    agg = (
        obj
        .groupby(["player_name", "date"])[num_cols]
        .agg(["mean", "max"])
    )

    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    print("Daily aggregated rows:", len(agg))
    return agg

def impute(df):
    for c in df.columns:
        if c in ["player_name", "date", "label"]:
            continue
        df[c] = df.groupby("player_name")[c].transform(lambda x: x.fillna(x.mean()))

    df = df.fillna(df.median(numeric_only=True))
    return df

def build_labels(obj_daily, injuries):
    obj_daily = obj_daily.sort_values(["player_name", "date"])
    injuries = injuries.sort_values("timestamp")

    injury_map = {}

    for _, row in injuries.iterrows():
        pid = row["player_name"]
        d = row["timestamp"].date()
        injury_map.setdefault(pid, []).append(d)

    labels = []

    for _, row in obj_daily.iterrows():
        pid = row["player_name"]
        d = row["date"]
        label = 0

        if pid in injury_map:
            for inj_date in injury_map[pid]:
                delta = (inj_date - d).days
                if 0 <= delta <= FUTURE_DAYS:
                    label = 1
                    break

        labels.append(label)

    obj_daily["label"] = labels

    print("Positive samples:", obj_daily["label"].sum())
    print("Positive rate:", round(obj_daily["label"].mean(), 4))

    return obj_daily

def run_model(df):
    df = df.sort_values(["player_name", "date"])
    feature_cols = [c for c in df.columns if c not in ["player_name", "date", "label"]]

    X = df[feature_cols]
    y = df["label"]

    if y.nunique() < 2:
        print("Only one class present, model training aborted")
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
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, zero_division=0))

def main():
    injuries = load_subjective()
    obj = load_objective()
    obj_daily = aggregate_daily(obj)

    start_date = obj_daily["date"].min()
    end_date = obj_daily["date"].max()

    injuries = injuries[
        (injuries["timestamp"].dt.date >= start_date) &
        (injuries["timestamp"].dt.date <= end_date)
    ]

    print("Injuries overlapping objective window:", len(injuries))

    obj_daily = build_labels(obj_daily, injuries)
    obj_daily = impute(obj_daily)

    print("Final dataset shape:", obj_daily.shape)

    run_model(obj_daily)

if __name__ == "__main__":
    main()
