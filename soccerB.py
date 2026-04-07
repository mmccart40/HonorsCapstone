import pandas as pd
import numpy as np
import glob
import os

# =========================
# 1. LOAD OBJECTIVE DATA
# =========================

objective_files = glob.glob("/scratch/user/u.mm342941/objective-TeamB-2020/**/*.parquet", recursive=True)

objective_list = []

for file in objective_files:
    print(f"Loading objective: {file}")

    df = pd.read_parquet(file)

    # ---- FIX: timestamp → date ----
    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date

    # ---- CLEAN player names ----
    df["player_name"] = df["player_name"].astype(str).str.strip()

    objective_list.append(df)

objective = pd.concat(objective_list, ignore_index=True)

print("\nOBJECTIVE RAW SHAPE:", objective.shape)


# =========================
# 2. AGGREGATE OBJECTIVE TO PLAYER-DAY
# =========================

# Adjust feature names if needed
objective_daily = (
    objective
    .groupby(["player_name", "date"])
    .agg({
        # replace these with actual columns in your dataset
        "speed": "mean",
        "distance": "sum",
        "acceleration": "mean"
    })
    .reset_index()
)

print("\nOBJECTIVE DAILY SHAPE:", objective_daily.shape)


# =========================
# 3. LOAD SUBJECTIVE DATA
# =========================

subjective_files = glob.glob("/scratch/user/u.mm342941/subjective-TeamB/**/*.parquet", recursive=True)

subjective_list = []

for file in subjective_files:
    print(f"Loading subjective: {file}")

    df = pd.read_parquet(file)

    # ---- FIX PLAYER ID MISMATCH ----
    df["player_name"] = (
        df["player_name"]
        .astype(str)
        .str.replace(r"^TeamB-TeamA-", "TeamB-", regex=True)
        .str.strip()
    )

    # ---- FIX DATE FORMAT ----
    df["date"] = pd.to_datetime(
        df["timestamp"],
        format="%d.%m.%Y",
        errors="coerce"
    ).dt.date

    subjective_list.append(df)

subjective = pd.concat(subjective_list, ignore_index=True)

print("\nSUBJECTIVE RAW SHAPE:", subjective.shape)


# =========================
# 4. AGGREGATE SUBJECTIVE TO PLAYER-DAY
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

print("\nSUBJECTIVE DAILY SHAPE:", subjective_daily.shape)


# =========================
# 5. ALIGN DATE RANGE
# =========================

common_start = max(objective_daily["date"].min(), subjective_daily["date"].min())
common_end = min(objective_daily["date"].max(), subjective_daily["date"].max())

print("\nCOMMON DATE RANGE:", common_start, "→", common_end)

objective_daily = objective_daily[
    objective_daily["date"].between(common_start, common_end)
]

subjective_daily = subjective_daily[
    subjective_daily["date"].between(common_start, common_end)
]


# =========================
# 6. MERGE
# =========================

final_df = objective_daily.merge(
    subjective_daily,
    on=["player_name", "date"],
    how="left"
)

print("\nFINAL SHAPE:", final_df.shape)


# =========================
# 7. DEBUG CHECKS
# =========================

print("\n===== MERGE COVERAGE =====")
print("team_performance coverage:",
      final_df["team_performance"].notna().mean())

print("\n===== SAMPLE OUTPUT =====")
print(final_df.head(10))

print("\n===== NULL CHECK =====")
print(final_df.isna().mean())