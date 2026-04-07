import pandas as pd
import numpy as np
import glob

# =========================
# HELPERS
# =========================

def find_timestamp_column(df):
    candidates = [
        "timestamp", "time", "datetime",
        "date_time", "event_time", "eventtime"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def safe_to_date(df, col):
    return pd.to_datetime(df[col], errors="coerce").dt.date


# =========================
# LOAD OBJECTIVE DATA
# =========================

objective_files = glob.glob(
    "/scratch/user/u.mm342941/objective-TeamB-2020/**/*.parquet",
    recursive=True
)

objective_list = []
bad_obj_files = 0

for file in objective_files:
    print("[OBJECTIVE] Loading:", file)

    try:
        df = pd.read_parquet(file)

        ts_col = find_timestamp_column(df)

        if ts_col is None:
            print("SKIP: no timestamp column")
            bad_obj_files += 1
            continue

        df["date"] = safe_to_date(df, ts_col)
        df = df.dropna(subset=["date"])

        if df.empty:
            print("SKIP: empty after date parsing")
            bad_obj_files += 1
            continue

        if "player_name" not in df.columns:
            print("SKIP: no player_name column")
            bad_obj_files += 1
            continue

        df["player_name"] = df["player_name"].astype(str).str.strip()

        objective_list.append(df)

    except Exception as e:
        print("FAILED FILE:", file)
        print("ERROR:", e)
        bad_obj_files += 1
        continue


objective = pd.concat(objective_list, ignore_index=True)

print("\nOBJECTIVE RAW SHAPE:", objective.shape)
print("BAD OBJECTIVE FILES:", bad_obj_files)


# =========================
# AGGREGATE OBJECTIVE TO PLAYER-DAY
# =========================

numeric_cols = objective.select_dtypes(include=[np.number]).columns.tolist()
agg_dict = {c: "mean" for c in numeric_cols}

objective_daily = (
    objective
    .groupby(["player_name", "date"])
    .agg(agg_dict)
    .reset_index()
)

print("OBJECTIVE DAILY SHAPE:", objective_daily.shape)


# =========================
# LOAD SUBJECTIVE DATA
# =========================

subjective_files = glob.glob(
    "/scratch/user/u.mm342941/subjective-TeamB/**/*.parquet",
    recursive=True
)

subjective_list = []
bad_sub_files = 0

for file in subjective_files:
    print("[SUBJECTIVE] Loading:", file)

    try:
        df = pd.read_parquet(file)

        if "player_name" not in df.columns:
            print("SKIP: no player_name")
            bad_sub_files += 1
            continue

        df["player_name"] = (
            df["player_name"]
            .astype(str)
            .str.replace(r"^TeamB-TeamA-", "TeamB-", regex=True)
            .str.strip()
        )

        ts_col = find_timestamp_column(df)

        if ts_col is None:
            print("SKIP: no timestamp column")
            bad_sub_files += 1
            continue

        df["date"] = safe_to_date(df, ts_col)
        df = df.dropna(subset=["date"])

        if df.empty:
            print("SKIP: empty after date parsing")
            bad_sub_files += 1
            continue

        subjective_list.append(df)

    except Exception as e:
        print("FAILED SUBJECTIVE FILE:", file)
        print("ERROR:", e)
        bad_sub_files += 1
        continue


subjective = pd.concat(subjective_list, ignore_index=True)

print("\nSUBJECTIVE RAW SHAPE:", subjective.shape)
print("BAD SUBJECTIVE FILES:", bad_sub_files)


# =========================
# AGGREGATE SUBJECTIVE TO PLAYER-DAY
# =========================

subjective_daily = (
    subjective
    .groupby(["player_name", "date"])
    .agg({
        "team_performance": "mean",
        "offensive_performance": "mean",
        "defensive_performance": "mean",
        "problems": "first"
    })
    .reset_index()
)

print("SUBJECTIVE DAILY SHAPE:", subjective_daily.shape)


# =========================
# ALIGN DATE RANGE
# =========================

common_start = max(objective_daily["date"].min(), subjective_daily["date"].min())
common_end = min(objective_daily["date"].max(), subjective_daily["date"].max())

print("\nCOMMON DATE RANGE:", common_start, "->", common_end)

objective_daily = objective_daily[
    objective_daily["date"].between(common_start, common_end)
]

subjective_daily = subjective_daily[
    subjective_daily["date"].between(common_start, common_end)
]


# =========================
# MERGE
# =========================

final_df = objective_daily.merge(
    subjective_daily,
    on=["player_name", "date"],
    how="left"
)

print("\nFINAL SHAPE:", final_df.shape)


# =========================
# DEBUG CHECKS
# =========================

print("\nMERGE COVERAGE:")
print(final_df["team_performance"].notna().mean())

print("\nSAMPLE:")
print(final_df.head(10))

print("\nNULL RATE:")
print(final_df.isna().mean().sort_values(ascending=False))

print("\nBAD OBJECTIVE FILES:", bad_obj_files)
print("BAD SUBJECTIVE FILES:", bad_sub_files)