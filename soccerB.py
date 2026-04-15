import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OBJECTIVE_ROOT = "/scratch/user/u.mm342941/objective-TeamA-2020"
SUBJECTIVE_ROOT = os.path.join(BASE_DIR, "subjective")

MAX_FILES = 200  # keep small for debugging


# ---------------- SUBJECTIVE LOADING (SAFE) ----------------
def load_subjective():
    injury_path = os.path.join(SUBJECTIVE_ROOT, "injury", "injury.csv")
    illness_path = os.path.join(SUBJECTIVE_ROOT, "illness", "illness.csv")

    def safe_read(path):
        if not os.path.exists(path):
            print("Missing subjective file:", path)
            return None
        return pd.read_csv(path)

    injury = safe_read(injury_path)
    illness = safe_read(illness_path)

    if injury is None:
        raise ValueError("No injury file found")

    injury["timestamp"] = pd.to_datetime(
        injury["timestamp"], format="%d.%m.%Y", errors="coerce"
    )

    injury = injury.dropna(subset=["timestamp"])

    print("\nSUBJECTIVE SUMMARY")
    print("Injury rows:", len(injury))
    print("Unique players:", injury["player_name"].nunique())
    print("Date range:", injury["timestamp"].min(), "to", injury["timestamp"].max())

    return injury


# ---------------- OBJECTIVE DIAGNOSTICS ----------------
def inspect_parquet_structure(files):
    column_counter = Counter()
    sample_schemas = []
    timestamp_candidates = Counter()

    valid_files = 0

    for i, f in enumerate(files[:MAX_FILES]):
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print("READ ERROR:", f, e)
            continue

        valid_files += 1

        cols = list(df.columns)
        column_counter.update(cols)

        sample_schemas.append(cols)

        # detect possible timestamp columns
        for c in cols:
            if "time" in c.lower() or "date" in c.lower():
                timestamp_candidates[c] += 1

        if i < 3:
            print("\nSAMPLE FILE:", f)
            print("Columns:", cols)
            print(df.head(2))

    print("\nOBJECTIVE STRUCTURE SUMMARY")
    print("Valid files read:", valid_files)

    print("\nMost common columns:")
    for k, v in column_counter.most_common(20):
        print(k, ":", v)

    print("\nPossible timestamp columns:")
    for k, v in timestamp_candidates.most_common():
        print(k, ":", v)

    return column_counter, timestamp_candidates


# ---------------- OBJECTIVE LOADING (NON-DESTRUCTIVE) ----------------
def load_objective_debug():
    files = glob.glob(os.path.join(OBJECTIVE_ROOT, "**/*.parquet"), recursive=True)

    print("\nTOTAL PARQUET FILES FOUND:", len(files))

    column_counter, timestamp_candidates = inspect_parquet_structure(files)

    return files, column_counter, timestamp_candidates


# ---------------- CHECK IF ANY USABLE DATA EXISTS ----------------
def try_extract_any_objective(files):
    usable_rows = []
    schema_fail = 0

    for i, f in enumerate(files[:MAX_FILES]):
        try:
            df = pd.read_parquet(f)
        except:
            continue

        # try to normalize timestamp
        ts_col = None
        for c in df.columns:
            if c.lower() in ["timestamp", "time", "datetime", "date"]:
                ts_col = c
                break

        if ts_col is None:
            schema_fail += 1
            continue

        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        # detect player column
        player_col = None
        for c in df.columns:
            if "player" in c.lower() or "athlete" in c.lower():
                player_col = c
                break

        if player_col is None:
            continue

        df = df.rename(columns={player_col: "player_name"})

        if "speed" not in df.columns and "heart_rate" not in df.columns:
            continue

        df["date"] = df["timestamp"].dt.date

        usable_rows.append(df[["player_name", "date"]])

    if len(usable_rows) == 0:
        print("\nNO USABLE OBJECTIVE DATA FOUND")
        print("Files missing timestamp schema:", schema_fail)
        return None

    obj = pd.concat(usable_rows, ignore_index=True)

    print("\nPOTENTIAL OBJECTIVE DATA FOUND")
    print("Rows:", len(obj))
    print("Players:", obj["player_name"].nunique())
    print("Date range:", obj["date"].min(), "to", obj["date"].max())

    return obj


# ---------------- OVERLAP CHECK ----------------
def check_overlap(obj, injuries):
    if obj is None:
        print("\nCANNOT CHECK OVERLAP: NO OBJECTIVE DATA")
        return

    obj_dates = set(obj["date"].unique())
    inj_dates = set(injuries["timestamp"].dt.date.unique())

    overlap = obj_dates.intersection(inj_dates)

    print("\nOVERLAP ANALYSIS")
    print("Objective unique dates:", len(obj_dates))
    print("Injury unique dates:", len(inj_dates))
    print("Overlapping dates:", len(overlap))

    if len(overlap) == 0:
        print("\nCRITICAL ISSUE: NO TEMPORAL OVERLAP")
        print("This means model training is impossible with current split.")
    else:
        print("Overlap exists → modeling possible")


# ---------------- MAIN ----------------
def main():
    injuries = load_subjective()

    files, col_count, ts_candidates = load_objective_debug()

    obj = try_extract_any_objective(files)

    check_overlap(obj, injuries)

    print("\nDIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()