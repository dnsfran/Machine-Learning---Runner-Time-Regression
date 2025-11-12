from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r'.\courses_2020_fixed_distance_km_no_top5.csv'

def encode_gender(g):
    if pd.isna(g):
        return np.nan
    g_s = str(g).strip().upper()
    if g_s.startswith('M'):
        return 1
    if g_s.startswith('F'):
        return 0
    return np.nan


current_year=2020
df = pd.read_csv(path)

# age
if 'Athlete year of birth' in df.columns:
    df['age'] = current_year - df['Athlete year of birth']
else:
    df['age'] = np.nan

# gender code
if 'Athlete gender' in df.columns:
    df['gender_code'] = df['Athlete gender'].apply(encode_gender)
else:
    df['gender_code'] = np.nan

X = df[['age', 'gender_code', 'Distance_km']].copy()
y = df['Athlete average speed'].copy()

mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# simple hyperparameter grid
n_estimators_list = [50, 100, 200]
max_depth_list = [None, 5, 10]

results = []
for n in n_estimators_list:
    for md in max_depth_list:
        rf = RandomForestRegressor(n_estimators=n, max_depth=md)
        rf.fit(X_train_scaled, y_train)

        y_pred_train = rf.predict(X_train_scaled)
        y_pred_val = rf.predict(X_val_scaled)
        y_pred_test = rf.predict(X_test_scaled)

        tr_mse = mean_squared_error(y_train, y_pred_train)
        val_mse = mean_squared_error(y_val, y_pred_val)
        test_mse = mean_squared_error(y_test, y_pred_test)

        results.append({
            'n_estimators': n,
            'max_depth': md,
            'train_mse': tr_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
        })

for md in max_depth_list:
    n_vals = []
    val_mses = []
    test_mses = []
    train_mses = []
    for res in results:
        if res['max_depth'] == md:
            n_vals.append(res['n_estimators'])
            train_mses.append(res['train_mse'])
            val_mses.append(res['val_mse'])
            test_mses.append(res['test_mse'])
    plt.figure()
    plt.plot(n_vals, train_mses, label='Train MSE', marker='o')
    plt.plot(n_vals, val_mses, label='Validation MSE', marker='o')
    plt.plot(n_vals, test_mses, label='Test MSE', marker='o')
    plt.xlabel('n_estimators')
    plt.ylabel('MSE')
    plt.title(f'MSE vs n_estimators (max_depth={md})')
    plt.legend()
    plt.grid(True)
    plt.show()



