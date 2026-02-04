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
    'days_since_last_injury_start'
]

df_norm = df_filled.copy()

# Normalize using Min-Max per athlete so that variables are on a common
# scale of 0-1. Differences in athletes do not dominate correlations

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
corr_df = df_norm[wellness_vars + training_load_vars]
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

# Define target variable
target = 'injury_in_next_7d' # 0 = no injury, 1 = injury

# Define features
features = wellness_vars + training_load_vars
X = df_norm[features]
y = df_norm[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Baseline logistic regression
log_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='liblinear'
)
log_model.fit(X_train, y_train)

y_prob_log = log_model.predict_proba(X_test)[:, 1]
threshold = 0.30
y_pred_log = (y_prob_log >= threshold).astype(int)


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

# Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)
cm_dt = confusion_matrix(y_test, y_pred_dt)

print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log, zero_division=0))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))


def evaluate(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }

results = {
    'Logistic Regression': evaluate(y_test, y_pred_log),
    'Decision Tree': evaluate(y_test, y_pred_dt)
}

results_df = pd.DataFrame(results).T
print(results_df)
