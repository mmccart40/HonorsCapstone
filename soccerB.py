# ----------------------------
# LOAD SUBJECTIVE DATA (SAFE)
# ----------------------------
print("Loading subjective data...\n")

def load_if_exists(path, name):
    if os.path.exists(path):
        print(f"Loaded: {path}")
        df = pd.read_csv(path)

        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(" ", "_")
        df.columns = df.columns.str.replace(".", "", regex=False)

        df["source"] = name
        return df
    else:
        print(f"Missing: {path}")
        return None

subjective_dfs = []

files_to_check = [
    ("subjective/injury/injury.csv", "injury"),
    ("subjective/wellness/wellness.csv", "wellness"),
    ("subjective/training-load/training-load.csv", "training"),
    ("subjective/game-performance/game-performance.csv", "game"),
    ("subjective/illness/illness.csv", "illness"),
]

for path, name in files_to_check:
    df_temp = load_if_exists(path, name)
    if df_temp is not None:
        subjective_dfs.append(df_temp)

# Combine only what exists
if len(subjective_dfs) > 0:
    subjective_df = pd.concat(subjective_dfs, ignore_index=True)
else:
    print("No subjective data found")
    subjective_df = pd.DataFrame()

print("\nLoaded subjective datasets:", len(subjective_dfs))