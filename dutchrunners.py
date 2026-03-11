import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
# For correlation matrix
import matplotlib.pyplot as plt

#Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# LOAD THE DATASET
# Each athlete is identified by athlete_id
df = pd.read_csv('day_approach_maskedID_timeseries.csv')

# Formatting to make it easier to work with
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace(".", "", regex=False)

# IMPUTATE MISSING VALUES
# Missing values are done on a per-athlete basis
filled_dfs = []

for athlete_id in df['athlete_id'].unique():
    # Subset data for a single athlete
    athlete_df = df[df['athlete_id']==athlete_id].copy()

    # Forward fill and then back fill
    athlete_df = athlete_df.ffill().bfill()
    filled_dfs.append(athlete_df)

# Combine all athletes back into a single DataFrame
# reset index for duplicated indices
df_filled = pd.concat(filled_dfs).reset_index(drop=True)

# NORMALIZE THE DATA
cols_to_normalize = [
    'nr_sessions', 'total_km', 'km_z3-4', 'km_z5-t1-t2', 'km_sprinting',
    'strength_training', 'hours_alternative', 'perceived_exertion',
    'perceived_trainingsuccess', 'perceived_recovery'
]

scaler = MinMaxScaler()
df_filled[cols_to_normalize] = scaler.fit_transform(df_filled[cols_to_normalize])

# FEATURE SELECTION
target = 'injury_in_next_7d'  # 0 = no injury, 1 = injury

wellness_vars = [
    'perceived_exertion',
    'perceived_trainingsuccess',
    'perceived_recovery'
]

training_load_vars = [
    'nr_sessions',
    'total_km',
    'km_z3-4',
    'km_z5-t1-t2',
    'km_sprinting',
    'strength_training',
    'hours_alternative'
]

# FEATURE ENGINEERING
# Create Injury in the Next 7 Days Target Variable
df_filled['injury_in_next_7d'] = (
    df_filled.groupby('athlete_id')['injury']
    .shift(-7) # Look 7 days into the future for each athlete
    .fillna(0) # If we don't have 7 days of future data, assume no injury (could also choose to drop these rows instead)
    .astype(int)
)

# For each feature, compute rolling stats and ACWR per athlete
all_cols = training_load_vars + wellness_vars

for col in all_cols:

    # Build up result columns by looping through each athlete individually
    avg3_list  = []
    avg7_list  = []
    avg14_list = []
    std3_list  = []
    std7_list  = []
    std14_list = []
    trend7_list = []
    acwr_list  = []

    for athlete_id in df_filled['athlete_id'].unique():

        # Grab just this athlete's rows
        mask        = df_filled['athlete_id'] == athlete_id
        athlete_col = df_filled.loc[mask, col]

        # Rolling averages (smooth out daily noise)
        avg3  = athlete_col.rolling(window=3,  min_periods=1).mean()
        avg7  = athlete_col.rolling(window=7,  min_periods=1).mean()
        avg14 = athlete_col.rolling(window=14, min_periods=1).mean()

        # Rolling standard deviation (captures inconsistency/volatility)
        std3  = athlete_col.rolling(window=3,  min_periods=1).std().fillna(0)
        std7  = athlete_col.rolling(window=7,  min_periods=1).std().fillna(0)
        std14 = athlete_col.rolling(window=14, min_periods=1).std().fillna(0)

        # 7-day trend: today's value minus the value from 7 days ago
        # Positive = load going up, Negative = load going down
        trend7 = athlete_col.diff(7).fillna(0)

        # Acute:Chronic Workload Ratio
        # Acute = average load over last 7 days  (short term fatigue)
        # Chronic = average load over last 28 days (long term fitness)
        # Ratio > 1.3 means athlete is doing a lot more than they're used to = injury risk
        acute   = athlete_col.rolling(window=7,  min_periods=1).mean()
        chronic = athlete_col.rolling(window=28, min_periods=1).mean()
        acwr    = (acute / (chronic + 1e-6)).clip(0, 3)

        avg3_list.append(avg3)
        avg7_list.append(avg7)
        avg14_list.append(avg14)
        std3_list.append(std3)
        std7_list.append(std7)
        std14_list.append(std14)
        trend7_list.append(trend7)
        acwr_list.append(acwr)

    # Stitch all athletes back together and assign to df_filled
    df_filled[f'{col}_avg3']   = pd.concat(avg3_list)
    df_filled[f'{col}_avg7']   = pd.concat(avg7_list)
    df_filled[f'{col}_avg14']  = pd.concat(avg14_list)
    df_filled[f'{col}_std3']   = pd.concat(std3_list)
    df_filled[f'{col}_std7']   = pd.concat(std7_list)
    df_filled[f'{col}_std14']  = pd.concat(std14_list)
    df_filled[f'{col}_trend7'] = pd.concat(trend7_list)
    df_filled[f'{col}_acwr']   = pd.concat(acwr_list)

# Collect all feature columns: original + engineered
feature_prefixes = wellness_vars + training_load_vars
features = [
    col for col in df_filled.columns
    if any(col.startswith(p) for p in feature_prefixes)
]

# Drop any rows where features or target couldn't be filled
df_filled = df_filled.dropna(subset=features + [target])

# Normalise the newly engineered features
engineered_cols = [c for c in features if c not in cols_to_normalize]
scaler_eng = MinMaxScaler()
df_filled[engineered_cols] = scaler_eng.fit_transform(df_filled[engineered_cols])

print(f"Total features: {len(features)}")

# BUILD SEQUENCES

# Number of days the model will look back to make a prediction about injury in the next 7 days
SEQUENCE_LENGTH = 7
X_sequences = []
y_sequences = []

for athlete_id in df_filled['athlete_id'].unique():
    athlete_df = df_filled[df_filled['athlete_id'] == athlete_id]
    X_athlete  = athlete_df[features].values
    y_athlete  = athlete_df[target].values

    for i in range(len(athlete_df) - SEQUENCE_LENGTH):
        X_sequences.append(X_athlete[i:i + SEQUENCE_LENGTH])
        y_sequences.append(y_athlete[i + SEQUENCE_LENGTH])

# Convert to numpy arrays
X_sequences = np.array(X_sequences, dtype=np.float32) # shape = (num_samples, SEQUENCE_LENGTH, num_features)
y_sequences = np.array(y_sequences, dtype=np.float32)

print(f"Class balance --> no-injury: {(y_sequences==0).sum()}  |  injury: {(y_sequences==1).sum()}")
print(f"Injury rate --> {y_sequences.mean()*100:.2f}%\n")

# TRAINING / TEST SPLIT
# Split into training and testing sets
# Reserve 20% of the data for testing, and stratify to maintain class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.2,
    stratify=y_sequences,
    random_state=42
)

# Flatten for non-sequence models (RF, LR)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# OVERSAMPLING
print("Applying manual oversampling...")

# Separate injury and non-injury rows
injury_rows = X_train_flat[y_train == 1]
injury_labels = y_train[y_train == 1]
no_injury_rows = X_train_flat[y_train == 0]
no_injury_labels = y_train[y_train == 0]

# Figure out how many extra injury rows we need
num_to_add = len(no_injury_rows) - len(injury_rows)

# Randomly pick from existing injury rows (with replacement)
rng = np.random.default_rng(seed=42)
random_idx = rng.choice(len(injury_rows), size=num_to_add, replace=True)
extra_rows = injury_rows[random_idx]
extra_labels = injury_labels[random_idx]

# Stack everything together
X_train_sm = np.vstack([X_train_flat, extra_rows])
y_train_sm = np.concatenate([y_train, extra_labels])

print(f"After oversampling --> no-injury: {(y_train_sm==0).sum()}  |  injury: {(y_train_sm==1).sum()}\n")

# PyTorch tensors
X_train_t      = torch.FloatTensor(X_train)
X_test_t       = torch.FloatTensor(X_test)
X_train_flat_t = torch.FloatTensor(X_train_flat)   # ← add this line
X_test_flat_t  = torch.FloatTensor(X_test_flat)

# Oversampled tensors for neural nets
X_train_sm_t = torch.FloatTensor(X_train_sm)
y_train_sm_t = torch.FloatTensor(y_train_sm).unsqueeze(1)

y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

# Positive-class weight for BCEWithLogitsLoss
'''
pos_count = (y_train == 1).sum()
neg_count = (y_train == 0).sum()
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
print(f"pos_weight = {pos_weight.item():.2f}  (applied to both models)\n")
'''
pos_weight = torch.tensor([3.0], dtype=torch.float32)
print(f"pos_weight = {pos_weight.item():.2f}  \n")

# RANDOM FOREST
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sm, y_train_sm)

# LOGISTIC REGRESSION
print("Training Logistic Regression...")
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_sm, y_train_sm)

# LSTM MODEL
class InjuryLSTM(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden,
            num_layers=layers, 
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)          # raw logits — BCEWithLogitsLoss handles sigmoid
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

lstm_pos_weight = torch.tensor([10.0], dtype=torch.float32)

num_features = X_train.shape[2]
lstm_model   = InjuryLSTM(num_features)
loss_fn_lstm = nn.BCEWithLogitsLoss(pos_weight=lstm_pos_weight)
opt_lstm     = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler    = optim.lr_scheduler.StepLR(opt_lstm, step_size=20, gamma=0.5)

y_train_t = torch.FloatTensor(y_train).unsqueeze(1)

# Train LSTM
print("\nTraining LSTM...")
for epoch in range(100):
    lstm_model.train()
    logits = lstm_model(X_train_t)
    loss   = loss_fn_lstm(logits, y_train_t)
    opt_lstm.zero_grad()
    loss.backward()
    opt_lstm.step()
    scheduler.step()
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}  loss={loss.item():.4f}")

# Simple Feedforward Neural Network
num_flat = X_train_flat.shape[1]

class InjuryNN(nn.Module):
    def __init__(self, input_size, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)           # raw logits
        )

    def forward(self, x):
        return self.net(x)

nn_model = InjuryNN(num_flat)
loss_fn_nn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt_nn = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-4)
sched_nn = optim.lr_scheduler.StepLR(opt_nn, step_size=50, gamma=0.5)

print("\nTraining Feedforward NN...")
for epoch in range(150):
    nn_model.train()
    logits = nn_model(X_train_sm_t)
    loss   = loss_fn_nn(logits, y_train_sm_t)
    opt_nn.zero_grad()
    loss.backward()
    opt_nn.step()
    sched_nn.step()
    if epoch % 30 == 0:
        print(f"  Epoch {epoch:3d}  loss={loss.item():.4f}")

# THRESHOLD SELECTION
def best_threshold_nn(model, X_t, y_true):
    model.eval()
    with torch.no_grad():
        raw   = model(X_t).numpy().flatten()
        probs = 1 / (1 + np.exp(-raw))  # convert logits to probabilities

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def best_threshold_sklearn(model, X, y_true):
    probs = model.predict_proba(X)[:, 1]

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

lstm_threshold = best_threshold_nn(lstm_model, X_train_t,      y_train)
nn_threshold   = best_threshold_nn(nn_model,   X_train_flat_t, y_train)
rf_threshold   = best_threshold_sklearn(rf,       X_train_flat,   y_train)
lr_threshold   = best_threshold_sklearn(lr_model, X_train_flat,   y_train)

print(f"\nThresholds --> LSTM: {lstm_threshold:.2f} | NN: {nn_threshold:.2f} | RF: {rf_threshold:.2f} | LR: {lr_threshold:.2f}\n")

# EVALUATION FUNCTIONS
def evaluate_nn(model, X_t, y_true, threshold):
    model.eval()
    with torch.no_grad():
        raw   = model(X_t).numpy().flatten()
        probs = 1 / (1 + np.exp(-raw))
    preds = (probs >= threshold).astype(int)
    return {
        'Accuracy': accuracy_score(y_true, preds),
        'Precision': precision_score(y_true, preds, zero_division=0),
        'Recall': recall_score(y_true, preds, zero_division=0),
        'F1': f1_score(y_true, preds, zero_division=0),
    }

def evaluate_sklearn(model, X, y_true, threshold):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        'Accuracy': accuracy_score(y_true, preds),
        'Precision': precision_score(y_true, preds, zero_division=0),
        'Recall': recall_score(y_true, preds, zero_division=0),
        'F1': f1_score(y_true, preds, zero_division=0),
    }

lstm_results = evaluate_nn(lstm_model, X_test_t,      y_test, lstm_threshold)
nn_results = evaluate_nn(nn_model,   X_test_flat_t, y_test, nn_threshold)
rf_results = evaluate_sklearn(rf,       X_test_flat, y_test, rf_threshold)
lr_results = evaluate_sklearn(lr_model, X_test_flat, y_test, lr_threshold)

results_df = pd.DataFrame({
    "LSTM": lstm_results,
    "Simple NN": nn_results,
    "Random Forest": rf_results,
    "Logistic Reg": lr_results,
}).T

results_df = results_df[['Accuracy', 'Precision', 'Recall', 'F1']].round(4)

print("======= MODEL PERFORMANCE =======")
print(results_df.to_string())

# FEATURE IMPORTANCE (Random Forest)
flat_feature_names = []
for t in range(SEQUENCE_LENGTH):
    for feat in features:
        flat_feature_names.append(f"{feat}_t{t}")

importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 6))
plt.barh(
    [flat_feature_names[i] for i in top_idx[::-1]],
    importances[top_idx[::-1]],
    color='steelblue'
)
plt.xlabel('Importance')
plt.title('Top 20 Features — Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("\nFeature importance chart saved --> feature_importance.png")