from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


csv_path = r'.\courses_2020_fixed_distance_km_no_top5.csv'

df = pd.read_csv(csv_path)

# compute age if year of birth present, otherwise leave NaN
current_year = 2020
if 'Athlete year of birth' in df.columns:
    df['age'] = current_year - df['Athlete year of birth']
else:
    df['age'] = np.nan

# encode gender: male=1, female=0, unknown->nan
def encode_gender(g):
    if pd.isna(g):
        return np.nan
    g_s = str(g).strip().upper()
    if g_s.startswith('M'):
        return 1
    if g_s.startswith('F'):
        return 0
    return np.nan

df['gender_code'] = df['Athlete gender'].apply(encode_gender)


X = df[['age', 'gender_code', 'Distance_km']].copy()
y = df['Athlete average speed'].copy()

mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

# Now split for training/validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50)

## define a list of values for the maximum polynomial degree 
degrees = list(range(1, 6))    

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

# scaler sur X_train uniquement (pour éviter fuite de données)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ensuite tu fais ton PolynomialFeatures




linear_tr_errors = []          
linear_val_errors = []
linear_test_errors = []

for degree in degrees:    # use for-loop to fit polynomial regression models with different degrees
    #TRAINING
    lin_regr = LinearRegression(fit_intercept=True) 
    poly = PolynomialFeatures(degree=degree, include_bias=False)    # generate polynomial features
    X_train_poly = poly.fit_transform(X_train_scaled)    # fit the raw features
    lin_regr.fit(X_train_poly, y_train)    # apply linear regression to these new features and labels

    #TRAINING ERROR
    y_pred_train = lin_regr.predict(X_train_poly)    # predict using the linear model
    tr_error = mean_squared_error(y_train, y_pred_train)    # calculate the training error
    
    #VALIDATION ERROR
    X_val_poly = poly.transform(X_val_scaled) # transform the raw features for the validation data 
    y_pred_val = lin_regr.predict(X_val_poly) # predict values for the validation data using the linear model 
    val_error = mean_squared_error(y_val, y_pred_val) # calculate the validation error
    
    #TEST ERROR
    X_test_poly = poly.transform(X_test_scaled) # transform the raw features for the test data
    y_pred_test = lin_regr.predict(X_test_poly) # predict values for the test data using the linear model
    test_error = mean_squared_error(y_test, y_pred_test) # calculate the test error

    #PLOTS
    linear_tr_errors.append(tr_error)
    linear_val_errors.append(val_error)
    linear_test_errors.append(test_error)


# plot the training and validation errors
plt.figure(figsize=(8, 5))
plt.plot(degrees, linear_tr_errors, label='Training Error', marker='o')
plt.plot(degrees, linear_val_errors, label='Validation Error', marker='o')
plt.plot(degrees, linear_test_errors, label='Test Error', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Training, Validation and Test Errors vs Polynomial Degree')
plt.legend()
plt.grid(True)
plt.show()