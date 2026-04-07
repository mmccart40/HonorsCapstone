import os
import gc
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
DATA_FOLDER = "/scratch/user/u.mm342941/objective-TeamB-2020"
SUBJECTIVE_FOLDER = "subjective"
MAX_FILES = 50

# ----------------------------
# LOAD SUBJECTIVE DATA
# ----------------------------
print("Loading subjective data...\n")

def load_subjective(file_path, name):
    df = pd.read_csv(file_path)

    # clean column names
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace(".", "", regex=False)

    df["source"] = name
    return df

injury_df   = load_subjective(f"{SUBJECTIVE_FOLDER}/injury/injury.csv", "injury")
wellness_df = load_subjective(f"{SUBJECTIVE_FOLDER}/wellness/wellness.csv", "wellness")
training_df = load_subjective(f"{SUBJECTIVE_FOLDER}/training-load/training-load.csv", "training")

# Combine all subjective data
subjective_df = pd.concat([injury_df, wellness_df, training_df], ignore_index=True)

print("Subjective columns:", subjective_df.columns.tolist())

# ----------------------------
# STANDARDIZE KEYS
# ----------------------------

# Rename athlete_id -> player_name so it matches GPS
if "athlete_id" in subjective_df.columns:
    subjective_df = subjective_df.rename(columns={"athlete_id": "player_name"})

# Ensure date format matches
if "date" in subjective_df.columns:
    subjective_df["date"] = pd.to_datetime(subjective_df["date"], errors="coerce").dt.date

# ----------------------------
# COLLECT OBJECTIVE FILES
# ----------------------------
files = []

for root, dirs, filenames in os.walk(DATA_FOLDER):
    for f in filenames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))

files.sort()

print(f"\nFound {len(files)} parquet files")
print(f"Processing first {MAX_FILES} files...\n")

all_data = []
bad_files = []

# ----------------------------
# PROCESS FUNCTION
# ----------------------------
def process_file(file_path):

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"SKIP: {file_path}")
        return None

    keep_cols = [
        'player_name', 'time', 'lat', 'lon', 'speed', 'heart_rate',
        'hacc', 'hdop', 'signal_quality', 'num_satellites',
        'inst_acc_impulse', 'accl_x', 'accl_y', 'accl_z',
        'gyro_x', 'gyro_y', 'gyro_z'
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

    # ------------------------
    # DATE (CRITICAL FIX)
    # ------------------------
    filename = os.path.basename(file_path)
    df["date"] = pd.to_datetime(filename[:10], errors="coerce").dt.date

    # ------------------------
    # SORT
    # ------------------------
    if "player_name" in df.columns and "time" in df.columns:
        df = df.sort_values(["player_name", "time"])

    # ------------------------
    # IMPUTE
    # ------------------------
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    if "player_name" in df.columns and len(numeric_cols) > 0:
        df[numeric_cols] = (
            df.groupby("player_name")[numeric_cols]
            .transform(lambda x: x.ffill().bfill())
        )

    return df


# ----------------------------
# MAIN LOOP
# ----------------------------
for i, file_path in enumerate(files[:MAX_FILES]):

    print(f"[{i+1}/{MAX_FILES}] {file_path}")

    df = process_file(file_path)

    if df is None:
        bad_files.append(file_path)
        continue

    # sample to reduce memory
    all_data.append(df.head(500))

    del df
    gc.collect()


# ----------------------------
# COMBINE OBJECTIVE DATA
# ----------------------------
print("\nCombining GPS data...\n")

if len(all_data) > 0:
    final_df = pd.concat(all_data, ignore_index=True)
else:
    print("No valid data")
    exit()

# ----------------------------
# 🔥 MERGE WITH SUBJECTIVE DATA
# ----------------------------
print("\nMerging subjective data...\n")

# Merge on player + date
final_df = final_df.merge(
    subjective_df,
    on=["player_name", "date"],
    how="left"
)

# ----------------------------
# CREATE INJURY TARGET
# ----------------------------
# depends on column naming — print to inspect
print("\nChecking injury columns...")
print([c for c in final_df.columns if "injury" in c])

# Example: adjust if needed
if "injury" in final_df.columns:
    final_df["injury"] = final_df["injury"].fillna(0)

# ----------------------------
# OUTPUT SAMPLE
# ----------------------------
print("\n===== FINAL DATA SAMPLE =====")
print(final_df.head(30))

print("\nSHAPE:", final_df.shape)
print("\nCOLUMNS:", list(final_df.columns))

# ----------------------------
# ONE ATHLETE OVER TIME
# ----------------------------
sample_player = final_df["player_name"].iloc[0]
player_df = final_df[final_df["player_name"] == sample_player]

print("\n===== ONE ATHLETE WITH INJURY DATA =====")
print("Athlete:", sample_player)
print(player_df.head(30))

# ----------------------------
# SUMMARY
# ----------------------------
print("\nDONE")
print("Bad files:", len(bad_files))