#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 14:37:01 2025

@author: tremampeloquio
FML HW2
"""
#%%
#load data

import pandas as pd

# Load the dataset
file_path = "techSalaries2017.csv"
df = pd.read_csv(file_path)

# Display basic info and first few rows to understand the dataset
df.info(), df.head()

#%%
#Q1

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# Select relevant numeric predictor variables excluding salary components
predictor_columns = [
    "yearsofexperience", "yearsatcompany", "Masters_Degree", "Bachelors_Degree",
    "Doctorate_Degree", "Highschool", "Some_College", "Race_Asian", "Race_White",
    "Race_Two_Or_More", "Race_Black", "Race_Hispanic", "Age", "Height", "Zodiac", "SAT", "GPA"
]

# Drop rows with missing values in predictors or target variable
df_filtered = df[["totalyearlycompensation"] + predictor_columns].dropna()

# Define X (predictors) and y (target variable)
X = df_filtered[predictor_columns]
y = df_filtered["totalyearlycompensation"]

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and compute R-squared for full model
y_pred = model.predict(X_test)
r2_full = r2_score(y_test, y_pred)

# Find best single predictor
best_r2 = 0
best_predictor = None

for col in predictor_columns:
    model_single = LinearRegression()
    X_train_single = X_train[[col]]
    X_test_single = X_test[[col]]
    
    model_single.fit(X_train_single, y_train)
    y_pred_single = model_single.predict(X_test_single)
    r2_single = r2_score(y_test, y_pred_single)
    
    if r2_single > best_r2:
        best_r2 = r2_single
        best_predictor = col

best_predictor, best_r2, r2_full

#scatterplot
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot with regression line
plt.figure(figsize=(8,6))
sns.regplot(x=X_test["yearsofexperience"], y=y_test, scatter_kws={'alpha':0.3}, line_kws={"color": "red"})
plt.xlabel("Years of Relevant Experience")
plt.ylabel("Total Annual Compensation ($)")
plt.title("Relationship Between Experience and Compensation")
plt.show()

#bar chart with r^2 values of full model vs best predictor
plt.figure(figsize=(6,4))
plt.bar(["Best Predictor (Experience)", "Full Model"], [best_r2, r2_full], color=["blue", "green"])
plt.ylabel("R-Squared Value")
plt.title("Variance Explained by Experience vs. Full Model")
plt.ylim(0, 1)
plt.show()


#%%
#Q2

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score


# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train OLS model for comparison
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)
y_pred_ols = ols_model.predict(X_test)
r2_ols = r2_score(y_test, y_pred_ols)

# Perform Ridge Regression with cross-validation
alphas = np.logspace(-2, 3, 100)  # Search over a wide range of lambda values
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid={'alpha': alphas}, scoring='r2', cv=5)
ridge_cv.fit(X_train, y_train)

# Train Ridge model with optimal alpha
best_alpha = ridge_cv.best_params_['alpha']
ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_train, y_train)
y_pred_ridge = ridge_best.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Compare results
print(f"OLS R^2: {r2_ols:.4f}")
print(f"Ridge R^2 (best alpha={best_alpha:.4f}): {r2_ridge:.4f}")


# Ridge Coefficient Shrinkage Plot
alphas = np.logspace(-2, 3, 100)
coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

plt.figure(figsize=(8,6))
plt.plot(alphas, coefs)
plt.xscale("log")
plt.xlabel("Lambda (Alpha)")
plt.ylabel("Coefficient Values")
plt.title("Ridge Coefficient Shrinkage")
plt.legend(X.columns, loc="best", fontsize="small", frameon=False)
plt.show()

#scatterplot

# Selecting relevant features (assuming these exist in the dataset)
features = ['yearsofexperience', 'yearsatcompany','Age', 'GPA', 'SAT']
target = 'totalyearlycompensation'

# Drop rows with missing values in selected columns
df = df.dropna(subset=features + [target])

# Define X and y
X = df[features]
y = df[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit OLS model
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)

# Fit Ridge model with optimal alpha
optimal_alpha = 15.1991
ridge = Ridge(alpha=optimal_alpha)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ols, alpha=0.5, label="OLS Predictions", color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Reference line
plt.xlabel("Actual Compensation")
plt.ylabel("Predicted Compensation")
plt.title("OLS Regression: Actual vs. Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, alpha=0.5, label="Ridge Predictions", color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Reference line
plt.xlabel("Actual Compensation")
plt.ylabel("Predicted Compensation")
plt.title("Ridge Regression: Actual vs. Predicted")

plt.tight_layout()
plt.show()

#%%
#Q3

from sklearn.linear_model import LassoCV

# Use the same features as in previous regression models
features = [
    "yearsofexperience", "yearsatcompany", "Masters_Degree", "Bachelors_Degree",
    "Doctorate_Degree", "Highschool", "Some_College", "Race_Asian", "Race_White",
    "Race_Two_Or_More", "Race_Black", "Race_Hispanic", "Age", "Height", "Zodiac", "SAT", "GPA"
]

target = "totalyearlycompensation"

# Drop rows with missing values in selected columns
df_filtered = df.dropna(subset=features + [target])

# Define X and y
X = df_filtered[features]
y = df_filtered[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform Lasso regression with cross-validation to find the best alpha
lasso_cv = LassoCV(alphas=np.logspace(-2, 3, 100), cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

# Get the optimal alpha value
best_alpha = lasso_cv.alpha_

# Fit Lasso model with optimal alpha
lasso_best = LassoCV(alphas=[best_alpha], cv=5).fit(X_train, y_train)
y_pred_lasso = lasso_best.predict(X_test)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Count number of coefficients shrunk to zero
num_zero_coefs = np.sum(lasso_best.coef_ == 0)

# Print results
print(f"Optimal lambda for Lasso: {best_alpha:.4f}")
print(f"Lasso R^2: {r2_lasso:.4f}")
print(f"Number of coefficients shrunk to zero: {num_zero_coefs}")

# Plot coefficient shrinkage
plt.figure(figsize=(8,6))
plt.bar(features, lasso_best.coef_, color=["red" if coef == 0 else "blue" for coef in lasso_best.coef_])
plt.xticks(rotation=90)
plt.ylabel("Coefficient Value")
plt.title("Lasso Regression: Coefficient Shrinkage")
plt.show()

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_lasso, alpha=0.5, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Compensation")
plt.ylabel("Predicted Compensation")
plt.title("Lasso Regression: Actual vs. Predicted")
plt.show()
#%%
#Q4

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("techSalaries2017.csv")

# Keep relevant columns and drop missing values
df_filtered = df.dropna(subset=["totalyearlycompensation", "gender"])

# Convert gender to binary (Male = 1, Female = 0, excluding non-binary or missing values)
df_filtered = df_filtered[df_filtered["gender"].isin(["Male", "Female"])]
df_filtered["Gender_Binary"] = df_filtered["gender"].apply(lambda x: 1 if x == "Male" else 0)

# Define features and target variable
X = df_filtered[["totalyearlycompensation"]]
y = df_filtered["Gender_Binary"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Female", "Male"], yticklabels=["Female", "Male"])
plt.xlabel("Predicted Gender")
plt.ylabel("Actual Gender")
plt.title("Confusion Matrix: Predicting Gender from Compensation")
plt.show()
#%%
#Q5

# Keep relevant columns and drop missing values
df_filtered = df.dropna(subset=["yearsofexperience", "Age", "Height", "SAT", "GPA", "totalyearlycompensation"])

# Create high vs. low earner binary outcome variable based on median salary
median_salary = df_filtered["totalyearlycompensation"].median()
df_filtered["High_Earner"] = (df_filtered["totalyearlycompensation"] > median_salary).astype(int)

# Define predictors and outcome variable
features = ["yearsofexperience", "Age", "Height", "SAT", "GPA"]
X = df_filtered[features]
y = df_filtered["High_Earner"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print accuracy and coefficients
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Feature Coefficients:")
for feature, coef in zip(features, log_reg.coef_[0]):
    print(f"{feature}: {coef:.4f}")
    

# Calculate sensitivity, specificity, and precision
TP = conf_matrix[1, 1]  # True Positives
TN = conf_matrix[0, 0]  # True Negatives
FP = conf_matrix[0, 1]  # False Positives
FN = conf_matrix[1, 0]  # False Negatives

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# Print accuracy, sensitivity, specificity, and precision
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")

# Visualize confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Low Earner", "High Earner"], yticklabels=["Low Earner", "High Earner"])
plt.xlabel("Predicted Income Level")
plt.ylabel("Actual Income Level")
plt.title("Confusion Matrix: Predicting High vs. Low Earners")
plt.show()

#%%
#Extra Credit A

from scipy.stats import shapiro

variables = {"Salary": "totalyearlycompensation", "Height": "Height", "Age": "Age"}

for name, col in variables.items():
    data = df_filtered[col]
    stat, p_value = shapiro(data)
    print(f"Shapiro-Wilk Test for {name}: W={stat:.4f}, p={p_value:.4f}")
    
    plt.figure(figsize=(6,4))
    sns.histplot(data, bins=30, kde=True)
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {name}")
    plt.show()
    
    if p_value < 0.05:
        print(f"{name} is NOT normally distributed (p < 0.05).")
    else:
        print(f"{name} appears to be normally distributed (p >= 0.05).")
        
#%%
#Extra Credit B

df_filtered = df.dropna(subset=["yearsofexperience", "Age", "Height", "SAT", "GPA", "totalyearlycompensation"])

#Violin plot for salary distribution
plt.figure(figsize=(10,6))
sns.violinplot(x=df_filtered["title"], y=df_filtered["totalyearlycompensation"], inner="quartile")
plt.xticks(rotation=45)
plt.xlabel("Job Title")
plt.ylabel("Total Yearly Compensation")
plt.title("Salary Distribution by Job Title")
plt.show()

#Remove extreme outliers (keeping data within 1.5 * IQR)
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

df_filtered = remove_outliers(df_filtered, "totalyearlycompensation")


#Violin plot for salary distribution
plt.figure(figsize=(10,6))
sns.violinplot(x=df_filtered["title"], y=df_filtered["totalyearlycompensation"], inner="quartile")
plt.xticks(rotation=45)
plt.xlabel("Job Title")
plt.ylabel("Total Yearly Compensation")
plt.title("Salary Distribution by Job Title")
plt.show()


