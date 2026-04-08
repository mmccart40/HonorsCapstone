import os
import gc
import pandas as pd

DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"
SUBJECTIVE_FOLDER = "subjective"
MAX_FILES = 50


# ----------------------------
# HELPERS
# ----------------------------
def standardize_player(x):
    x = str(x).strip()
    if x.startswith("TeamA-"):
        return x.replace("TeamA-", "TeamB-")
    if not x.startswith("TeamB-"):
        return f"TeamB-{x}"
    return x


def parse_date(series):
    return pd.to_datetime(series, format="%d.%m.%Y", errors="coerce").dt.date


# ----------------------------
# LOAD SUBJECTIVE
# ----------------------------
def load_subjective(name, path):
    if not os.path.exists(path):
        print("Missing:", path)
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    if "player_name" not in df.columns or "timestamp" not in df.columns:
        return None

    df["player_name"] = df["player_name"].apply(standardize_player)
    df["date"] = parse_date(df["timestamp"])

    df = df.dropna(subset=["player_name", "date"])

    return df


paths = [
    ("injury", f"{SUBJECTIVE_FOLDER}/injury/injury.csv"),
    ("performance", f"{SUBJECTIVE_FOLDER}/game-performance/game-performance.csv"),
    ("illness", f"{SUBJECTIVE_FOLDER}/illness/illness.csv"),
]

subjective_dfs = []

for name, path in paths:
    df = load_subjective(name, path)
    if df is not None:
        subjective_dfs.append(df)

subjective_all = pd.concat(subjective_dfs, ignore_index=True)

print("Subjective rows:", len(subjective_all))


# ----------------------------
# PROCESS OBJECTIVE → DAILY AGGREGATION
# ----------------------------
def process_file(file_path):
    try:
        df = pd.read_parquet(file_path)
    except:
        return None

    if "player_name" not in df.columns:
        return None

    df["player_name"] = df["player_name"].apply(standardize_player)

    filename = os.path.basename(file_path)
    date_val = pd.to_datetime(filename[:10], errors="coerce")

    if pd.isna(date_val):
        return None

    df["date"] = date_val.date()

    # KEEP ONLY NEEDED COLS
    keep_cols = ["player_name", "date", "speed", "heart_rate", "accl_x", "accl_y", "accl_z"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # ----------------------------
    # KEY FIX: AGGREGATE TO DAILY
    # ----------------------------
    agg_df = df.groupby(["player_name", "date"]).agg({
        "speed": ["mean", "max"],
        "heart_rate": ["mean", "max"],
        "accl_x": "mean",
        "accl_y": "mean",
        "accl_z": "mean"
    })

    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns]
    agg_df = agg_df.reset_index()

    return agg_df


# ----------------------------
# MAIN LOOP
# ----------------------------
files = []

for root, _, filenames in os.walk(DATA_FOLDER):
    for f in filenames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))

files.sort()

all_data = []

for i, file in enumerate(files[:MAX_FILES]):
    print(f"[{i+1}/{MAX_FILES}] Processing")

    df = process_file(file)

    if df is not None:
        all_data.append(df)

    gc.collect()


# ----------------------------
# COMBINE
# ----------------------------
final_df = pd.concat(all_data, ignore_index=True)

print("Final shape:", final_df.shape)


# ----------------------------
# MERGE (NOW IT WORKS)
# ----------------------------
final_df = final_df.merge(
    subjective_all,
    on=["player_name", "date"],
    how="left"
)


# ----------------------------
# CHECK
# ----------------------------
print("\nMerge coverage:")
print(final_df[subjective_all.columns].notna().mean())

print("\nSample merged rows:")
print(final_df.dropna(subset=subjective_all.columns).head())


print("\nDone")