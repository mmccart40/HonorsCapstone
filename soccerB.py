import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

OBJECTIVE_PATH = "/scratch/user/u.mm342941/objective-TeamA-2020"
SUBJECTIVE_PATH = "subjective"

# -----------------------------
# LOAD SUBJECTIVE
# -----------------------------
def load_subjective():
    dfs = []

    files = [
        "injury/injury.csv",
        "game-performance/game-performance.csv",
        "illness/illness.csv"
    ]

    for path in files:
        full_path = os.path.join(SUBJECTIVE_PATH, path)

        if not os.path.exists(full_path):
            continue

        df = pd.read_csv(full_path)

        df["date"] = pd.to_datetime(
            df["timestamp"],
            format="%d.%m.%Y",
            errors="coerce"
        )

        df = df.dropna(subset=["date"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df

subjective_df = load_subjective()

# Keep only TeamA
subjective_df = subjective_df[
    subjective_df["player_name"].str.contains("TeamA")
]

team_players = set(subjective_df["player_name"])

# -----------------------------
# CREATE INJURY LABEL TABLE
# -----------------------------
injury_df = subjective_df[subjective_df["type"].notna()].copy()
injury_df = injury_df[["player_name", "date"]].drop_duplicates()

# -----------------------------
# LOAD OBJECTIVE
# -----------------------------
files = glob.glob(OBJECTIVE_PATH + "/**/*.parquet", recursive=True)
files = files[:800]

results = []

for i, f in enumerate(files):
    if (i + 1) % 50 == 0:
        print(f"[{i+1}/{len(files)}]")

    try:
        df = pd.read_parquet(f)

        if "player_name" not in df.columns:
            continue

        df = df[df["player_name"].isin(team_players)]
        if df.empty:
            continue

        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        elif "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
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

        results.append(agg)

    except Exception:
        continue

objective_df = pd.concat(results, ignore_index=True)

# -----------------------------
# CREATE LABEL: injury in next 7 days
# -----------------------------
objective_df = objective_df.sort_values(["player_name", "date"])
injury_df = injury_df.sort_values(["player_name", "date"])

objective_df["injury_next_7d"] = 0

injury_lookup = injury_df.groupby("player_name")["date"].apply(list).to_dict()

for idx, row in objective_df.iterrows():
    player = row["player_name"]
    date = row["date"]

    if player not in injury_lookup:
        continue

    future_dates = injury_lookup[player]

    for inj_date in future_dates:
        if 0 < (inj_date - date).days <= 7:
            objective_df.at[idx, "injury_next_7d"] = 1
            break

# -----------------------------
# FEATURE ENGINEERING (rolling)
# -----------------------------
objective_df = objective_df.sort_values(["player_name", "date"])

for col in ["speed_mean", "hr_mean"]:
    objective_df[f"{col}_7d_avg"] = (
        objective_df.groupby("player_name")[col]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

# -----------------------------
# PREPARE DATA
# -----------------------------
features = [
    "speed_mean", "speed_max",
    "hr_mean", "hr_max",
    "accl_x_mean", "accl_y_mean", "accl_z_mean",
    "speed_mean_7d_avg", "hr_mean_7d_avg"
]

df_model = objective_df.dropna(subset=features)

X = df_model[features]
y = df_model["injury_next_7d"]

print("Positive rate:", y.mean())

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

try:
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
except:
    print("ROC AUC could not be computed")

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df_model.to_csv("final_features.csv", index=False)

print("Done")