# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Data Preparation:** Load the car dataset, handle missing values, encode categorical features, and split data into training and testing sets.
2. **Model Training:** Fit a linear regression model using the training data with car price as the target variable.
3. **Prediction & Evaluation:** Predict car prices on the test set and evaluate performance using metrics such as R² and Mean Squared Error (MSE).
4. **Assumption Testing:** Check linear regression assumptions—linearity (scatter plots), independence (Durbin–Watson test), homoscedasticity (residual plots), normality of errors (Q–Q plot), and multicollinearity (VIF).
## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
df = pd.read_csv('CarPrice_Assignment.csv')
df.head()
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print('Name: SYED ADIL S ')
print('Reg. No: 212225040453')
print("MODEL COEFFICIENTS:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")
print("\nMODEL PERFORMANCE:")

print(f"{'MSE':>12}: {mean_squared_error(y_test, y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test, y_pred)):>10.2f}")
print(f"{'R.squared':>12}: {r2_score(y_test, y_pred):>10.2f}")
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price($)")
plt.ylabel("Predicted Price($)")
plt.grid(True)
plt.show()
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Warson Statistic: {dw_test:.2f}", 
      "\n(Values Close to 2 indicate no autocorrelation)")
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals-vs-Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
<img width="385" height="296" alt="image" src="https://github.com/user-attachments/assets/0d4abde3-c46f-4903-b3d9-faa61b542193" />
<img width="1335" height="607" alt="image" src="https://github.com/user-attachments/assets/6c4769cc-5ccc-4fb0-8fb0-3c46305da316" />
<img width="1320" height="663" alt="image" src="https://github.com/user-attachments/assets/d558263e-5e76-470c-bc40-ef857f0bc3c8" />
<img width="1343" height="515" alt="image" src="https://github.com/user-attachments/assets/60926a02-18d8-4869-b932-edcdf7af40ec" />


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
