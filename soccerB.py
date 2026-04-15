import os
import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

OBJECTIVE_ROOT = "/scratch/user/u.mm342941/objective-TeamA-2020"
SUBJECTIVE_PATH = "subjective"

FUTURE_DAYS = 7


def parse_date(df, col):
    df[col] = pd.to_datetime(df[col], format="%d.%m.%Y", errors="coerce")
    return df


def load_subjective():
    injury = pd.read_csv(f"{SUBJECTIVE_PATH}/injury/injury.csv")
    perf = pd.read_csv(f"{SUBJECTIVE_PATH}/game-performance/performance.csv")
    illness = pd.read_csv(f"{SUBJECTIVE_PATH}/illness/illness.csv")

    injury = parse_date(injury, "timestamp")
    perf = parse_date(perf, "timestamp")
    illness = parse_date(illness, "timestamp")

    injury["injury"] = 1
    perf["injury"] = 0
    illness["injury"] = 0

    injury = injury.rename(columns={"type": "injury_detail"})
    perf = perf.rename(columns={"team_performance": "team_perf"})

    df = pd.concat([
        injury[["player_name", "timestamp", "injury"]],
        perf[["player_name", "timestamp", "injury"]],
        illness[["player_name", "timestamp", "injury"]]
    ])

    df = df.dropna(subset=["timestamp"])

    print("Subjective rows:", len(df))
    print("Unique players:", df["player_name"].nunique())
    print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())

    return df


def load_objective():
    files = glob.glob(os.path.join(OBJECTIVE_ROOT, "**/*.parquet"), recursive=True)

    rows = []

    for i, f in enumerate(files[:800]):
        try:
            df = pd.read_parquet(f)

            if "timestamp" not in df.columns:
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["date"] = df["timestamp"].dt.date

            rows.append(df)

        except:
            continue

    if len(rows) == 0:
        raise ValueError("No valid objective data found")

    obj = pd.concat(rows, ignore_index=True)

    print("Objective rows:", len(obj))

    return obj


def aggregate_daily(obj):
    num_cols = ["speed", "heart_rate", "accl_x", "accl_y", "accl_z"]

    agg = obj.groupby(["player_name", "date"])[num_cols].agg(["mean", "max"])
    agg.columns = ["_".join(c) for c in agg.columns]

    agg = agg.reset_index()

    return agg


def impute(df):
    for c in df.columns:
        if c in ["player_name", "date"]:
            continue

        df[c] = df.groupby("player_name")[c].transform(
            lambda x: x.fillna(x.mean())
        )

    df = df.fillna(df.median(numeric_only=True))

    return df


def build_labels(obj_daily, injuries):
    injuries = injuries.sort_values("timestamp")

    obj_daily["label"] = 0

    injury_map = {}

    for _, row in injuries.iterrows():
        pid = row["player_name"]
        dt = row["timestamp"].date()

        if pid not in injury_map:
            injury_map[pid] = []

        injury_map[pid].append(dt)

    labels = []

    for _, row in obj_daily.iterrows():
        pid = row["player_name"]
        d = row["date"]

        label = 0

        if pid in injury_map:
            for inj in injury_map[pid]:
                if 0 <= (inj - d).days <= FUTURE_DAYS:
                    label = 1
                    break

        labels.append(label)

    obj_daily["label"] = labels

    print("Positive rate:", obj_daily["label"].mean())

    return obj_daily


def run_model(df):
    df = df.sort_values(["player_name", "date"])

    feature_cols = [c for c in df.columns if c not in ["player_name", "date", "label"]]

    X = df[feature_cols]
    y = df["label"]

    if y.nunique() < 2:
        print("ERROR: Only one class present. Cannot train model")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


def main():
    injuries = load_subjective()

    obj = load_objective()

    obj_daily = aggregate_daily(obj)

    obj_daily = impute(obj_daily)

    obj_daily = build_labels(obj_daily, injuries)

    print("Final shape:", obj_daily.shape)

    run_model(obj_daily)


if __name__ == "__main__":
    main()