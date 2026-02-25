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



# 1. Load the dataset. Each athlete is identified by athlete_id
df = pd.read_csv('toy_injury_timeseries.csv')

# 2. Handle missinf values (imputate)
# Missing values are done on a per-athlete basis
filled_dfs = []

for athlete_id in df['athlete_id'].unique():
    # Subset data for a single athlete
    athlete_df = df[df['athlete_id']==athlete_id].copy()

    # Forward fill and then back fill
    athlete_df = athlete_df.ffill().bfill()
    filled_dfs.append(athlete_df)



# Combine all athletes back into a single DataFrame
# reset index for duplicated indices (dont think theres any duplicates
# but just in case)
df_filled = pd.concat(filled_dfs).reset_index(drop=True)

#df_ffill = df.ffill()
#df_filled = df_ffill.bfill()

# 3. Normalizing the Data:

cols_to_normalize = [
    'daily_load',
    'rpe',
    'sleep_hours',
    'soreness',
    'stress',
    'acute_load_7d',
    'chronic_load_28d',
    'acwr',
    'monotony_7d',
    'poor_sleep_7d',
    'days_since_last_injury_start',
    'ac_diff',              
    'sleep_trend_7d'        
]

df_norm = df_filled.copy()

# -----------------------------
# Feature engineering (AFTER df_norm exists)
# -----------------------------

# Acute - chronic difference
df_norm['ac_diff'] = (
    df_norm['acute_load_7d'] - df_norm['chronic_load_28d']
)

# Load spike indicator
df_norm['load_spike'] = (
    df_norm['acute_load_7d'] >
    1.3 * df_norm['chronic_load_28d']
).astype(int)

# 7-day sleep trend (per athlete)
df_norm['sleep_trend_7d'] = (
    df_norm
    .groupby('athlete_id')['sleep_hours']
    .diff(7)
    .fillna(0)
)

# Make sure data is sorted correctly
df_norm = df_norm.sort_values(['athlete_id', 'date'])

grouped = df_norm.groupby('athlete_id')

# Rolling soreness averages
for window in [3, 7, 28]:
    df_norm[f'soreness_avg_{window}d'] = (
        grouped['soreness']
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# Rolling stress averages
for window in [3, 7, 28]:
    df_norm[f'stress_avg_{window}d'] = (
        grouped['stress']
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# Rolling sleep averages
for window in [3, 7, 28]:
    df_norm[f'sleep_avg_{window}d'] = (
        grouped['sleep_hours']
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# Rolling standard deviation (volatility)
for window in [7, 28]:
    df_norm[f'load_std_{window}d'] = (
        grouped['daily_load']
        .rolling(window=window, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

# Short-term vs long-term difference
df_norm['soreness_trend'] = (
    df_norm['soreness_avg_3d'] -
    df_norm['soreness_avg_28d']
)

df_norm['load_trend'] = (
    df_norm['acute_load_7d'] -
    df_norm['chronic_load_28d']
)

# Normalize using Min-Max per athlete so that variables are on a common
# scale of 0-1. Differences in athletes do not dominate correlations

'''
for athlete_id in df_norm['athlete_id'].unique():
    mask = df_norm['athlete_id'] == athlete_id
    
    for col in cols_to_normalize:
        min_val = df_norm.loc[mask, col].min()
        max_val = df_norm.loc[mask, col].max()
        
        # Avoid division by zero
        if max_val != min_val:
            df_norm.loc[mask, col] = (
                df_norm.loc[mask, col] - min_val
            ) / (max_val - min_val)
        else:
            df_norm.loc[mask, col] = 0
'''

wellness_vars = [
    'sleep_hours',
    'soreness',
    'stress',
    'poor_sleep_7d'
]

training_load_vars = [
    'daily_load',
    'acute_load_7d',
    'chronic_load_28d',
    'acwr',
    'monotony_7d'
]

# Create a subset using only variables we are interested in
'''
corr_df = df_norm[wellness_vars + training_load_vars]
corr_matrix = corr_df.corr()
'''



# Define target variable, binary target
target = 'injury_in_next_7d' # 0 = no injury, 1 = injury

# Define features
# Use selected wellness and training load variables as predictors
base_features = (
    wellness_vars +
    training_load_vars +
    ['ac_diff', 'load_spike', 'sleep_trend_7d']
)

memory_features = [
    col for col in df_norm.columns
    if 'avg_' in col or 'std_' in col or 'trend' in col
]

features = list(set(base_features + memory_features))
print("Total features:", len(features))
print("Unique features:", len(set(features)))

X = df_norm[features]
y = df_norm[target]

# Stratified split preserves injury /  no-injury ratio in train
# and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y # So the training set and test set have same injury proportion
)


# Only scale numeric continuous variables
scaler = MinMaxScaler()

X_train = X_train.copy()
X_test = X_test.copy()

cols_to_scale = [col for col in cols_to_normalize if col in X_train.columns]

X_train.loc[:, cols_to_scale] = scaler.fit_transform(
    X_train[cols_to_scale]
)

X_test.loc[:, cols_to_scale] = scaler.transform(
    X_test[cols_to_scale]
)

print(X_train[cols_to_scale].describe())


corr_df = X_train[wellness_vars + training_load_vars]
corr_matrix = corr_df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
im = plt.imshow(corr_matrix, vmin=-1, vmax=1)
plt.colorbar(im, label='Correlation Coefficient')

plt.xticks(
    ticks=np.arange(len(corr_matrix.columns)),
    labels=corr_matrix.columns,
    rotation=90
)

plt.yticks(
    ticks=np.arange(len(corr_matrix.columns)),
    labels=corr_matrix.columns
)

# Add correlation values inside each cell
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        value = corr_matrix.iloc[j, i]
        plt.text(
            i, j,
            f"{value:.2f}",
            ha='center',
            va='center',
            color='white' if abs(value) > 0.5 else 'black',
            fontsize=8
        )


plt.title('Correlation Analysis: Wellness and Training Load Metrics')
plt.tight_layout()
plt.show()

# Baseline logistic regression
# - class_weight = 'balanced' to handle class imbalance
# - higher max_iter to ensure convergence
log_model = LogisticRegression(
    max_iter=2000,
    class_weight={0:1, 1:5.5},
    solver='liblinear',
    C=0.7
)


log_model.fit(X_train, y_train)

X_tensor = torch.FloatTensor(X_train[features].values)
num_features = X_tensor.shape[1]

#LOG MODEL TWO
# TODO: Change the 12 to read the number of features - DO NOT KEEP HARDCODED
'''
log_model2 = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid()
)

# Convert data to tensors
X_tensor = torch.FloatTensor(X_train[features].values)
y_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)

# Setup
loss_fn = nn.BCELoss()              # Binary cross-entropy — same as logistic regression
optimizer = optim.Adam(log_model2.parameters(), lr=0.001)

# Train
for epoch in range(2000):
    y_pred = log_model2(X_tensor)
    loss = loss_fn(y_pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()                  # This is backpropagation — computes gradients
    optimizer.step()                 # This updates the weights

    if epoch == epoch - 1:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
'''

#SIMPLE NEURAL NET
# TODO: Change the 12 to read the number of features - DO NOT KEEP HARDCODED
firstNN = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# Model A
model_A = nn.Sequential(
    nn.Linear(num_features, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Model B
model_B = nn.Sequential(
    nn.Linear(num_features, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Model C
model_C = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

model_D = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.4),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

model_E = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Convert data to tensors
X_tensor = torch.FloatTensor(X_train[features].values)
y_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)

pos_weight_value = 5.5/1
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Setup
#loss_fn = nn.BCELoss()              # Binary cross-entropy — same as logistic regression
optimizer = optim.Adam(firstNN.parameters(), lr=0.001)

# Train
for epoch in range(1000):
    y_pred = firstNN(X_tensor)
    loss = loss_fn(y_pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()                  # This is backpropagation — computes gradients
    optimizer.step()                 # This updates the weights

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")





# Predict probabilities for the positive class (injury = 1)
y_prob_log = log_model.predict_proba(X_test)[:, 1]
threshold = 0.40
y_pred_log = (y_prob_log >= threshold).astype(int)

# Evaluate model performance at several fixed thresholds
# to understand precision-recall tradeoffs
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

# Search for the threshold that maximizes F1 score
best_t = 0.4
best_f1 = 0

for t in np.arange(0.1, 0.9, 0.05):
    y_pred_t = (y_prob_log >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

# Decision tree
dt_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=15,
    min_samples_split=50,
    class_weight='balanced',
    criterion='entropy',
    random_state=42
)

dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=20,
    min_samples_split=60,
    max_features='sqrt',
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)


rf_model.fit(X_train, y_train)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
threshold = 0.30
y_pred_rf = (y_prob_rf >= threshold).astype(int)

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:")
print(cm_rf)


# Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)
cm_dt = confusion_matrix(y_test, y_pred_dt)

'''
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log, zero_division=0))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))
'''

def train_model(model):
    pos_weight_value = 6.5 / 1
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model

# Train all neural nets
model_A = train_model(model_A)
model_B = train_model(model_B)
model_C = train_model(model_C)
model_D = train_model(model_D)
model_E = train_model(model_E)

def evaluate(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }

def evaluate_nn(model, X_test, y_test, threshold=0.4):
    
    model.eval()  # very important (turns off dropout/batchnorm randomness)
    
    X_test_tensor = torch.FloatTensor(X_test.values)
    
    with torch.no_grad():
        y_prob = model(X_test_tensor).numpy().flatten()
    
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0)
    }

results = {
    'Logistic Regression': evaluate(y_test, y_pred_log),
    'Decision Tree': evaluate(y_test, y_pred_dt),
    'Random Forest': evaluate(y_test, y_pred_rf),
    'Simple NN': evaluate_nn(firstNN, X_test[features], y_test),
    'Model A': evaluate_nn(model_A, X_test[features], y_test),
    'Model B': evaluate_nn(model_B, X_test[features], y_test),
    'Model C': evaluate_nn(model_C, X_test[features], y_test),
    'Model D': evaluate_nn(model_D, X_test[features], y_test),
    'Model E': evaluate_nn(model_E, X_test[features], y_test, threshold=0.25),
}

results_df = pd.DataFrame(results).T
results_df = results_df[['Accuracy','Precision','Recall','F1']]
results_df = results_df.round(6)

print("\n======= ALL MODEL PERFORMANCE ===========\n")
print(results_df.to_string())
