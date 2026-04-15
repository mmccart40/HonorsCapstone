import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG  ← adjust paths here
# ─────────────────────────────────────────
# Point this at the ROOT of all objective data (TeamA + TeamB, 2020 + 2021).
# The script will recursively find every parquet file under it.
# If you only have TeamA-2020, keep that path but know labels will be sparse.
OBJECTIVE_ROOT  = "/scratch/user/u.mm342941/objective-TeamA-2020"
SUBJECTIVE_PATH = "subjective"
TARGET_TEAM     = "TeamA"       # filter prefix; set None to keep all players
MAX_FILES       = None          # None = load everything; set e.g. 2000 for testing
FUTURE_DAYS     = 7             # predict injury in next N days (forward-looking)
SEQUENCE_LENGTH = 7             # rolling look-back window for features
CHUNK_SIZE      = 100


# ─────────────────────────────────────────
# 1. LOAD SUBJECTIVE DATA
# ─────────────────────────────────────────
def load_subjective():
    """
    Returns a DataFrame with columns: player_name, date, has_injury
    'has_injury' = 1 on any day a player reported an injury.

    The subjective CSVs use timestamp format DD.MM.YYYY.
    """
    records = []

    def _read(path, label):
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            return None
        df = pd.read_csv(path)
        print(f"  Loaded {label}: {df.shape}")
        # normalise column names
        df.columns = df.columns.str.strip().str.lower()

        # find timestamp column
        ts_col = next((c for c in df.columns if "timestamp" in c or "date" in c), None)
        if ts_col is None:
            print(f"  [SKIP] no timestamp column in {label}")
            return None

        df["date"] = pd.to_datetime(df[ts_col], format="%d.%m.%Y", errors="coerce")
        df = df.dropna(subset=["player_name", "date"])
        return df

    injury_df = _read(f"{SUBJECTIVE_PATH}/injury/injury.csv",               "injury")
    perf_df   = _read(f"{SUBJECTIVE_PATH}/game-performance/game-performance.csv", "performance")
    illness_df= _read(f"{SUBJECTIVE_PATH}/illness/illness.csv",             "illness")

    all_dfs = [d for d in [injury_df, perf_df, illness_df] if d is not None]
    if not all_dfs:
        raise RuntimeError("No subjective CSVs could be loaded.")

    combined = pd.concat(all_dfs, ignore_index=True)

    if TARGET_TEAM:
        combined = combined[combined["player_name"].str.startswith(TARGET_TEAM)]

    # Mark days that have an *injury* report (not illness/performance)
    # injury_df rows have a "type" column; keep those as positive labels
    combined["has_injury"] = 0
    if injury_df is not None:
        inj = injury_df.copy()
        if TARGET_TEAM:
            inj = inj[inj["player_name"].str.startswith(TARGET_TEAM)]
        inj["has_injury"] = 1
        # Build a set of (player, date) pairs that are injury days
        injury_days = set(zip(inj["player_name"], inj["date"].dt.normalize()))
    else:
        injury_days = set()

    print(f"\nSubjective rows total : {len(combined)}")
    print(f"Unique players        : {combined['player_name'].nunique()}")
    print(f"Injury-day records    : {len(injury_days)}")
    return combined, injury_days


# ─────────────────────────────────────────
# 2. PROCESS OBJECTIVE DATA
# ─────────────────────────────────────────
def process_objective():
    """
    Reads all parquet files under OBJECTIVE_ROOT, filters to TARGET_TEAM,
    and aggregates per (player_name, date).

    Key fix vs original:
    - Uses pd.to_datetime with infer_datetime_format=True + utc=True
      to suppress the repeated UserWarning.
    - Loads ALL available years/months so date ranges overlap with
      the subjective data.
    """
    files = glob.glob(f"{OBJECTIVE_ROOT}/**/*.parquet", recursive=True)
    print(f"\nTotal parquet files found: {len(files)}")

    if MAX_FILES:
        files = files[:MAX_FILES]
        print(f"  (capped at {MAX_FILES})")

    results = []
    for i in range(0, len(files), CHUNK_SIZE):
        chunk = files[i : i + CHUNK_SIZE]
        chunk_dfs = []
        print(f"  [{min(i+CHUNK_SIZE, len(files))}/{len(files)}] Processing...")

        for f in chunk:
            try:
                df = pd.read_parquet(f)
                if "player_name" not in df.columns:
                    continue
                if TARGET_TEAM:
                    df = df[df["player_name"].str.startswith(TARGET_TEAM)]
                if len(df) == 0:
                    continue

                # ── FAST, WARNING-FREE datetime parsing ──────────────────
                if "time" in df.columns:
                    # Try numeric (Unix ms) first, then string
                    if pd.api.types.is_numeric_dtype(df["time"]):
                        df["date"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                    else:
                        df["date"] = pd.to_datetime(
                            df["time"], infer_datetime_format=True, errors="coerce"
                        )
                elif "timestamp" in df.columns:
                    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
                else:
                    continue

                df = df.dropna(subset=["date"])
                df["date"] = df["date"].dt.tz_localize(None).dt.normalize()

                # ── Aggregate to one row per player per day ───────────────
                agg_dict = {}
                for col in ["speed", "heart_rate", "accl_x", "accl_y", "accl_z"]:
                    if col in df.columns:
                        agg_dict[col] = ["mean", "max", "std"]

                if not agg_dict:
                    continue

                agg = df.groupby(["player_name", "date"]).agg(agg_dict)
                agg.columns = ["_".join(c) for c in agg.columns]
                agg = agg.reset_index()
                chunk_dfs.append(agg)

            except Exception as e:
                continue

        if chunk_dfs:
            results.append(pd.concat(chunk_dfs, ignore_index=True))

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)

    # If the same player+date appears in multiple files (split sessions), re-aggregate
    num_cols = [c for c in df.columns if c not in ("player_name", "date")]
    df = df.groupby(["player_name", "date"])[num_cols].mean().reset_index()

    print(f"Objective rows (player-days): {len(df)}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Players: {df['player_name'].nunique()}")
    return df


# ─────────────────────────────────────────
# 3. BUILD DAILY PLAYER GRID + LABELS
# ─────────────────────────────────────────
def build_labeled_grid(objective_df, injury_days):
    """
    For each player, build a continuous daily timeline spanning the objective
    data range. Merge in objective features, then create a FORWARD-LOOKING
    label: 'injury_in_next_Nd' = 1 if the player reported an injury in the
    next FUTURE_DAYS days.

    This mirrors the Dutch dataset's  shift(-7)  approach exactly.
    """
    all_players = objective_df["player_name"].unique()
    date_min    = objective_df["date"].min()
    date_max    = objective_df["date"].max()

    print(f"\nBuilding daily grid: {date_min.date()} → {date_max.date()} "
          f"for {len(all_players)} players")

    full_index_rows = []
    for player in all_players:
        dates = pd.date_range(start=date_min, end=date_max, freq="D")
        full_index_rows.append(
            pd.DataFrame({"player_name": player, "date": dates})
        )
    grid = pd.concat(full_index_rows, ignore_index=True)

    # Merge objective features
    grid = grid.merge(objective_df, on=["player_name", "date"], how="left")

    # Mark injury days (binary column per row)
    grid["injury_today"] = grid.apply(
        lambda r: 1 if (r["player_name"], r["date"]) in injury_days else 0,
        axis=1
    )

    print(f"Grid shape: {grid.shape}")
    print(f"Injury days in grid: {grid['injury_today'].sum()}")
    return grid


# ─────────────────────────────────────────
# 4. FEATURE ENGINEERING  (mirrors Dutch script)
# ─────────────────────────────────────────
def engineer_features(grid):
    """
    Per-player rolling features:
      - 3/7/14-day rolling mean, std
      - 7-day trend
      - ACWR  (acute 7d / chronic 28d)

    Forward-looking label:
      injury_in_next_Nd = 1 if any injury in the next FUTURE_DAYS days
    """
    # Determine which objective columns are available
    obj_cols = [c for c in grid.columns
                if c not in ("player_name", "date", "injury_today")
                and grid[c].dtype in [np.float64, np.float32, np.int64]]

    print(f"\nEngineering features for {len(obj_cols)} objective columns...")

    all_rows = []

    for player in grid["player_name"].unique():
        mask = grid["player_name"] == player
        pdf  = grid.loc[mask].copy().sort_values("date").reset_index(drop=True)

        # ── Forward-looking label (same logic as Dutch shift(-7)) ──────
        pdf["injury_in_next_Nd"] = (
            pdf["injury_today"]
            .rolling(window=FUTURE_DAYS, min_periods=1)
            .max()
            .shift(-FUTURE_DAYS)
            .fillna(0)
            .astype(int)
        )

        # ── Rolling features per objective column ───────────────────────
        for col in obj_cols:
            s = pdf[col]
            pdf[f"{col}_avg3"]   = s.rolling(3,  min_periods=1).mean()
            pdf[f"{col}_avg7"]   = s.rolling(7,  min_periods=1).mean()
            pdf[f"{col}_avg14"]  = s.rolling(14, min_periods=1).mean()
            pdf[f"{col}_std7"]   = s.rolling(7,  min_periods=1).std().fillna(0)
            pdf[f"{col}_trend7"] = s.diff(7).fillna(0)

            acute   = s.rolling(7,  min_periods=1).mean()
            chronic = s.rolling(28, min_periods=1).mean()
            pdf[f"{col}_acwr"]   = (acute / (chronic + 1e-6)).clip(0, 3)

        all_rows.append(pdf)

    df = pd.concat(all_rows, ignore_index=True)
    print(f"After feature engineering: {df.shape}")
    return df


# ─────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────
def train_and_evaluate(df):
    target   = "injury_in_next_Nd"
    drop_cols = {"player_name", "date", "injury_today", target}
    features  = [c for c in df.columns if c not in drop_cols
                 and df[c].dtype in [np.float64, np.float32, np.int64, np.float16]]

    df_model = df.dropna(subset=features + [target]).copy()

    print(f"\nModel dataset: {len(df_model)} rows, {len(features)} features")
    print(f"Positive rate: {df_model[target].mean():.4f}  "
          f"({int(df_model[target].sum())} injury windows / {len(df_model)} total)")

    if df_model[target].nunique() < 2:
        print("\n⚠  ERROR: Still only one class present after re-labeling.")
        print("   Possible causes:")
        print("   1. OBJECTIVE_ROOT only covers June 2020 — no injuries in that window.")
        print("      → Try pointing OBJECTIVE_ROOT at the full 2020+2021 data root.")
        print("   2. Player name format mismatch between parquet & CSV.")
        print("      → Run the diagnostics block at the bottom of this script.")
        return

    # Normalise
    scaler = MinMaxScaler()
    df_model[features] = scaler.fit_transform(df_model[features].fillna(0))

    X = df_model[features].values
    y = df_model[target].values

    # ── Oversampling (same as Dutch script) ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    inj_idx  = np.where(y_train == 1)[0]
    no_idx   = np.where(y_train == 0)[0]
    n_add    = len(no_idx) - len(inj_idx)
    if n_add > 0:
        rng     = np.random.default_rng(42)
        extra   = inj_idx[rng.choice(len(inj_idx), size=n_add, replace=True)]
        X_train = np.vstack([X_train, X_train[extra]])
        y_train = np.concatenate([y_train, y_train[extra]])
    print(f"After oversampling → no-injury: {(y_train==0).sum()} | injury: {(y_train==1).sum()}")

    # ── Models ─────────────────────────────────────────────────────────
    models = {
        "Random Forest":   RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boost":  HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=42
        ),
        "Logistic Reg":    LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree":   DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42
        ),
    }

    # SVM only on a sample (slow on large data)
    svm_size = min(10_000, len(X_train))
    rng2 = np.random.default_rng(99)
    svm_idx  = rng2.choice(len(X_train), size=svm_size, replace=False)
    models["SVM"] = SVC(
        kernel="rbf", class_weight="balanced", probability=True,
        C=1.0, max_iter=1000, random_state=42
    )

    results = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        if name == "SVM":
            model.fit(X_train[svm_idx], y_train[svm_idx])
        else:
            model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        results[name] = {
            "Accuracy":  accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall":    recall_score(y_test, preds, zero_division=0),
            "F1":        f1_score(y_test, preds, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_test, probs),
        }

    results_df = pd.DataFrame(results).T[
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    ].round(4)

    print("\n======= MODEL PERFORMANCE =======")
    print(results_df.to_string())

    # ── Feature importance (Random Forest) ─────────────────────────────
    rf = models["Random Forest"]
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [features[i] for i in top_idx[::-1]],
        importances[top_idx[::-1]],
        color="steelblue"
    )
    plt.xlabel("Importance")
    plt.title(f"Top 20 Features — Random Forest (predict injury in next {FUTURE_DAYS}d)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("\nFeature importance saved → feature_importance.png")

    return results_df


# ─────────────────────────────────────────
# DIAGNOSTICS  (run if positive rate = 0)
# ─────────────────────────────────────────
def run_diagnostics(objective_df, injury_days):
    """
    Prints date ranges and player name samples to help diagnose
    why objective and subjective data aren't overlapping.
    """
    print("\n====== DIAGNOSTICS ======")
    print(f"\nObjective date range : {objective_df['date'].min().date()} → {objective_df['date'].max().date()}")
    print(f"Objective players    : {sorted(objective_df['player_name'].unique())[:5]} ...")

    if injury_days:
        inj_dates   = sorted(d for _, d in injury_days)
        inj_players = sorted(set(p for p, _ in injury_days))
        print(f"\nInjury date range    : {inj_dates[0].date()} → {inj_dates[-1].date()}")
        print(f"Injured players      : {inj_players[:5]} ...")
    else:
        print("\nNo injury days found in subjective data.")

    overlap_players = (
        set(objective_df["player_name"].unique()) &
        set(p for p, _ in injury_days)
    )
    print(f"\nPlayer overlap count : {len(overlap_players)}")
    if overlap_players:
        print(f"Overlapping players  : {sorted(overlap_players)[:10]}")

    overlap_dates = set(objective_df["date"].dt.normalize())
    inj_date_set  = set(d for _, d in injury_days)
    overlap_d     = overlap_dates & inj_date_set
    print(f"\nDate overlap count   : {len(overlap_d)}")
    if overlap_d:
        print(f"Sample overlap dates : {sorted(overlap_d)[:5]}")
    else:
        print("  ⚠  ZERO date overlap — objective and injury data are in different time ranges.")
        print("     Solutions:")
        print("     a) Use the full 2020+2021 objective data root, not just TeamA-2020.")
        print("     b) Check if objective timestamps are in UTC and need timezone conversion.")
        print("     c) Confirm player_name format is identical in both sources.")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 50)
    print("SoccerMon Injury Prediction Pipeline")
    print("=" * 50)

    # 1. Subjective
    subjective_df, injury_days = load_subjective()

    # 2. Objective
    objective_df = process_objective()
    if objective_df.empty:
        print("No objective data loaded. Check OBJECTIVE_ROOT path.")
        return

    # 3. Diagnostics (always print — useful even when working)
    run_diagnostics(objective_df, injury_days)

    # 4. Build daily grid
    grid = build_labeled_grid(objective_df, injury_days)

    # 5. Feature engineering
    grid = engineer_features(grid)

    # 6. Train & evaluate
    train_and_evaluate(grid)


if __name__ == "__main__":
    main()