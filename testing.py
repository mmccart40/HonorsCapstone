import pandas as pd
# For correlation matrix
import matplotlib.pyplot as plt

df = pd.read_csv('toy_injury_timeseries.csv')

filled_dfs = []

for athlete_id in df['athlete_id'].unique():
    athlete_df = df[df['athlete_id']==athlete_id].copy()
    athlete_df = athlete_df.ffill().bfill()
    filled_dfs.append(athlete_df)

df_filled = pd.concat(filled_dfs).reset_index(drop=True)

#df_ffill = df.ffill()

#df_filled = df_ffill.bfill()

# Normalizing the Data:

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

for athlete_id in df_norm['athlete_id'].unique():
    mask = df_norm['athlete_id'] == athlete_id
    
    for col in cols_to_normalize:
        min_val = df_norm.loc[mask, col].min()
        max_val = df_norm.loc[mask, col].max()
        
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

corr_df = df_norm[wellness_vars + training_load_vars]
corr_matrix = corr_df.corr()

# Plot the correlation heatmap
plt.figure()
plt.imshow(corr_matrix)
plt.colorbar(label='Correlation Coefficient')

plt.xticks(
    ticks=np.arrange(len(corr_matrix.columns)),
    labels=corr_matrix.columns,
    rotation=90
)

plt.yticks(
    ticks=np.arrange(len(corr_matrix.columns)),
    labels=corr_matrix.columns
)

plt.title('Correlation Analysis: Wellness and Training Load Metrics')
plt.tight_layout()
plt.show()
