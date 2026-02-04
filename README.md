📌 Project Title

Delaney Solubility Prediction using Machine Learning

📍 Overview

This is my first Machine Learning project where I built regression models to predict the solubility of chemical compounds using molecular descriptors.

The dataset used is the Delaney Solubility Dataset, which contains various chemical features and the target value logS (log solubility).

🎯 Objective

The main goal of this project is:

✅ To predict the solubility (logS) of molecules
✅ To train and compare different regression models
✅ To evaluate model performance using MSE and R² score

📂 Dataset Information

Dataset Name: Delaney Solubility Dataset

Source: Data Professor GitHub Repository

Target Column: logS

Features: Molecular descriptors like weight, polarity, surface area, etc.

Dataset Link:

https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv

⚙️ Technologies Used

Python

Google Colab

Pandas

Scikit-learn

Matplotlib

Seaborn

🚀 Step-by-Step Workflow
1️⃣ Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

2️⃣ Load the Dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv"
)
df.head()


This loads the dataset containing chemical descriptors and solubility values.

3️⃣ Data Preparation
Separate Target Variable (Y)
y = df["logS"]

Separate Feature Variables (X)
X = df.drop("logS", axis=1)

4️⃣ Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)


80% data used for training

20% data used for testing

5️⃣ Model 1 — Linear Regression
Train Model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

Predictions
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

6️⃣ Model Evaluation

We use:

Mean Squared Error (MSE)

R² Score

from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

7️⃣ Model 2 — Random Forest Regression
Train Model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

Predictions
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

8️⃣ Random Forest Evaluation
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

9️⃣ Model Comparison
import pandas as pd

lr_results = pd.DataFrame(
    ["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]
).transpose()

rf_results = pd.DataFrame(
    ["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]
).transpose()

df_models = pd.concat([lr_results, rf_results])
df_models.columns = ["Model", "Train MSE", "Train R2", "Test MSE", "Test R2"]

df_models

🔟 Visualization of Predictions
plt.figure(figsize=(6,6))
plt.scatter(y_train, y_lr_train_pred, alpha=0.3)

plt.xlabel("Actual logS")
plt.ylabel("Predicted logS")
plt.title("Linear Regression Predictions")
plt.show()


This scatter plot shows how close predictions are to real values.

📊 Results
Model	Train R²	Test R²
Linear Regression	~0.76	~0.79
Random Forest	Depends on tuning	
✅ Conclusion

Linear Regression performed well as a baseline model.

Random Forest can improve performance with tuning.

This project demonstrates the complete ML pipeline from data loading to evaluation.
