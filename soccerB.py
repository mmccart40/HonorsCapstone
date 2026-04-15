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

FUTURE_DAYS = 7
MAX_PARQUET_FILES = 800


print("BASE_DIR:", BASE_DIR)
print("SUBJECTIVE_ROOT:", SUBJECTIVE_ROOT)
print("Exists:", os.path.exists(SUBJECTIVE_ROOT))


# ---------------- SUBJECTIVE LOADING ----------------
def safe_read_csv(path):
    if not os.path.exists(path):
        print("Missing file:", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print("Failed reading:", path, e)
        return None


def load_subjective():
    injury_path = os.path.join(SUBJECTIVE_ROOT, "injury", "injury.csv")

    # robust performance path search
    perf_candidates = [
        os.path.join(SUBJECTIVE_ROOT, "game-performance", "performance.csv"),
        os.path.join(SUBJECTIVE_ROOT, "performance", "performance.csv"),
        os.path.join(SUBJECTIVE_ROOT, "game-performance.csv"),
    ]

    illness_path = os.path.join(SUBJECTIVE_ROOT, "illness", "illness.csv")

    perf_path = next((p for p in perf_candidates if os.path.exists(p)), None)

    print("Loading subjective files")

    injury = safe_read_csv(injury_path)
    illness = safe_read_csv(illness_path)
    perf = safe_read_csv(perf_path) if perf_path else None

    if injury is None:
        raise ValueError("No injury file found")

    injury["timestamp"] = pd.to_datetime(injury["timestamp"], format="%d.%m.%Y", errors="coerce")
    injury["label"] = 1

    frames = [injury[["player_name", "timestamp", "label"]]]

    if illness is not None:
        illness["timestamp"] = pd.to_datetime(illness["timestamp"], format="%d.%m.%Y", errors="coerce")
        illness["label"] = 0
        frames.append(illness[["player_name", "timestamp", "label"]])

    if perf is not None:
        perf["timestamp"] = pd.to_datetime(perf["timestamp"], format="%d.%m.%Y", errors="coerce")
        perf["label"] = 0
        frames.append(perf[["player_name", "timestamp", "label"]])

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["timestamp"])

    print("Subjective rows:", len(df))
    print("Injury events:", df["label"].sum())

    return df[df["label"] == 1]


# ---------------- OBJECTIVE LOADING ----------------
def load_objective():
    files = glob.glob(os.path.join(OBJECTIVE_ROOT, "**/*.parquet"), recursive=True)

    print("Parquet files found:", len(files))

    rows = []

    for i, f in enumerate(files[:MAX_PARQUET_FILES]):
        if i % 100 == 0:
            print(f"[{i}/{len(files)}] Processing")

        try:
            df = pd.read_parquet(f)
        except Exception:
            continue

        if "player_name" not in df.columns or "timestamp" not in df.columns:
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        df["date"] = df["timestamp"].dt.date

        rows.append(df)

    if not rows:
        raise ValueError("No valid objective data loaded")

    obj = pd.concat(rows, ignore_index=True)

    print("Objective rows:", len(obj))
    print("Players:", obj["player_name"].nunique())

    return obj


# ---------------- DAILY AGG ----------------
def aggregate_daily(obj):
    metrics = ["speed", "heart_rate", "accl_x", "accl_y", "accl_z"]
    metrics = [m for m in metrics if m in obj.columns]

    if not metrics:
        raise ValueError("No numeric metrics found in objective data")

    daily = obj.groupby(["player_name", "date"])[metrics].agg(["mean", "max"])
    daily.columns = ["_".join(c) for c in daily.columns]
    daily = daily.reset_index()

    print("Daily rows:", len(daily))
    return daily


# ---------------- IMPUTATION ----------------
def impute(df):
    num_cols = df.select_dtypes(include=[np.number]).columns

    for c in num_cols:
        df[c] = df.groupby("player_name")[c].transform(lambda x: x.fillna(x.mean()))

    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    return df


# ---------------- LABELING ----------------
def build_labels(daily, injuries):
    injuries["date"] = injuries["timestamp"].dt.date

    injury_map = set(zip(injuries["player_name"], injuries["date"]))

    labels = []

    for _, row in daily.iterrows():
        pid = row["player_name"]
        d = row["date"]

        label = 0

        for k in range(FUTURE_DAYS):
            if (pid, d + pd.Timedelta(days=k).date()) in injury_map:
                label = 1
                break

        labels.append(label)

    daily["label"] = labels

    print("Positive rate:", daily["label"].mean())

    return daily


# ---------------- MODEL ----------------
def run_model(df):
    df = df.sort_values(["player_name", "date"])

    X = df.drop(columns=["player_name", "date", "label"])
    y = df["label"]

    if y.nunique() < 2:
        print("ERROR: Only one class present")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds, zero_division=0))


# ---------------- MAIN ----------------
def main():
    injuries = load_subjective()
    obj = load_objective()

    obj_daily = aggregate_daily(obj)

    # filter overlap safely
    injuries = injuries[
        (injuries["timestamp"].dt.date >= obj_daily["date"].min()) &
        (injuries["timestamp"].dt.date <= obj_daily["date"].max())
    ]

    print("Overlapping injuries:", len(injuries))

    obj_daily = build_labels(obj_daily, injuries)
    obj_daily = impute(obj_daily)

    print("Final shape:", obj_daily.shape)

    run_model(obj_daily)


if __name__ == "__main__":
    main()