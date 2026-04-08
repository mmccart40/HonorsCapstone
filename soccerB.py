import os
import gc
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"
SUBJECTIVE_FOLDER = "subjective"
MAX_FILES = 50
ROWS_PER_FILE = 200   # reduce memory usage

# ----------------------------
# SAFE DATE PARSER
# ----------------------------
def safe_parse_date(series):
    try:
        return pd.to_datetime(series, format="ISO8601", errors="coerce").dt.date
    except:
        return pd.to_datetime(series, errors="coerce").dt.date


# ----------------------------
# STANDARDIZE PLAYER NAME
# ----------------------------
def clean_player_name(x):
    x = str(x).strip()
    x = x.replace("TeamA-", "")
    if not x.startswith("TeamB-"):
        x = f"TeamB-{x}"
    return x


# ----------------------------
# LOAD SUBJECTIVE FILES
# ----------------------------
def load_subjective(path, name):

    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None

    print(f"\nLoading subjective: {name}")

    # robust CSV read
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip"
    )

    print("Raw shape:", df.shape)

    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    print("Columns:", df.columns.tolist())

    # ----------------------------
    # PLAYER NAME FIX
    # ----------------------------
    if "player_name" not in df.columns:
        print("No player_name column. Skipping.")
        return None

    df["player_name"] = df["player_name"].apply(clean_player_name)

    # ----------------------------
    # DATE FIX (CRITICAL)
    # ----------------------------
    if "timestamp" in df.columns:
        print("Parsing timestamp with %d.%m.%Y")
        df["date"] = pd.to_datetime(
            df["timestamp"].astype(str).str.strip(),
            format="%d.%m.%Y",
            errors="coerce"
        ).dt.date
    else:
        print("No timestamp column. Skipping.")
        return None

    print("Parsed date nulls:", df["date"].isna().sum(), "/", len(df))

    # ----------------------------
    # CLEAN
    # ----------------------------
    before = len(df)
    df = df.dropna(subset=["player_name", "date"])
    after = len(df)

    print(f"Rows after cleaning: {after} (dropped {before - after})")

    if after == 0:
        print("All rows dropped. Skipping dataset.")
        return None

    return df

# ----------------------------
# LOAD ALL SUBJECTIVE
# ----------------------------
print("Loading subjective data...")

paths = [
    ("injury", f"{SUBJECTIVE_FOLDER}/injury/injury.csv"),
    ("wellness", f"{SUBJECTIVE_FOLDER}/wellness/wellness.csv"),
    ("training", f"{SUBJECTIVE_FOLDER}/training-load/training-load.csv"),
    ("performance", f"{SUBJECTIVE_FOLDER}/game-performance/game-performance.csv"),
    ("illness", f"{SUBJECTIVE_FOLDER}/illness/illness.csv"),
]

subjective_dfs = []

for name, path in paths:
    df = load_subjective(path, name)
    if df is not None:
        subjective_dfs.append(df)

subjective_all = pd.concat(subjective_dfs, ignore_index=True) if subjective_dfs else None

if subjective_all is not None:
    print("Subjective rows:", len(subjective_all))
    print("Unique players:", subjective_all["player_name"].nunique())


# ----------------------------
# FIND PARQUET FILES
# ----------------------------
files = []

for root, dirs, filenames in os.walk(DATA_FOLDER):
    for f in filenames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))

files.sort()
print("Total parquet files:", len(files))


# ----------------------------
# PROCESS FILE
# ----------------------------
def process_file(file_path):

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print("Skipping file:", file_path)
        return None

    keep_cols = [
        "player_name", "time", "speed", "heart_rate",
        "accl_x", "accl_y", "accl_z"
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

    if "player_name" not in df.columns:
        return None

    df["player_name"] = df["player_name"].apply(clean_player_name)

    # extract date from filename (FAST + RELIABLE)
    filename = os.path.basename(file_path)
    date_val = pd.to_datetime(filename[:10], format="%Y-%m-%d", errors="coerce")

    if pd.isna(date_val):
        return None

    df["date"] = date_val.date()

    # reduce size early
    df = df.head(ROWS_PER_FILE)

    # convert time safely
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], format="ISO8601", errors="coerce")

    # numeric optimization
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # merge subjective
    if subjective_all is not None:
        df = df.merge(subjective_all, on=["player_name", "date"], how="left")

    return df


# ----------------------------
# MAIN LOOP
# ----------------------------
all_data = []
bad_files = []

for i, file_path in enumerate(files[:MAX_FILES]):

    print(f"[{i+1}/{MAX_FILES}] Processing")

    df = process_file(file_path)

    if df is None:
        bad_files.append(file_path)
        continue

    all_data.append(df)

    del df
    gc.collect()

print("\nOBJECTIVE DATE RANGE:", df["date"].min(), "to", df["date"].max())

if subjective_all is not None:
    print("SUBJECTIVE DATE RANGE:",
          subjective_all["date"].min(),
          "to",
          subjective_all["date"].max())

# ----------------------------
# FINAL COMBINE
# ----------------------------
print("Combining...")

if len(all_data) > 0:
    final_df = pd.concat(all_data, ignore_index=True)

    print("Final shape:", final_df.shape)
    print("Columns:", list(final_df.columns))

    if subjective_all is not None:
        merged_cols = [c for c in subjective_all.columns if c in final_df.columns]

        if merged_cols:
            coverage = final_df[merged_cols].notna().mean()
            print("\nMerge coverage:")
            print(coverage.sort_values(ascending=False).head(10))

else:
    print("No data processed")


# ----------------------------
# SUMMARY
# ----------------------------
print("Done")
print("Bad files:", len(bad_files))