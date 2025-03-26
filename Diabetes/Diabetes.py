#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:09:01 2025

@author: tremampeloquio
FML HW 3
"""

#import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

#%%

#load dataset
import pandas as pd

file_path = "diabetes.csv"
df = pd.read_csv(file_path)

df.info(), df.head()
#%%

#define feature set(x) and target variable (y)
X = df.drop(columns=["Diabetes"]) #features
y = df["Diabetes"] #target variable

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardize numerical features for models sensitive to scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#%%

import numpy as np
import matplotlib.pyplot as plt

#Q1: Logistic Regression

#train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_proba_log_reg = log_reg.predict_proba(X_test_scaled)[:,1]

#compute AUC
auc_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg)

#compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_log_reg)

#get best predictor
log_reg_importance = np.abs(log_reg.coef_).flatten()
best_feature_idx_log_reg = np.argmax(log_reg_importance)
best_feature_log_reg = X.columns[best_feature_idx_log_reg]

#display results
log_reg_results_df = pd.DataFrame({
    "Model": ["Logistic Regression"],
    "AUC Score": [auc_log_reg],
    "Best Predictor": [best_feature_log_reg]
    })

print(log_reg_results_df)

# plot  ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Logistic Regression (AUC = {auc_log_reg:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.grid()
plt.show()
#%%

#Q2: SVM

from sklearn.svm import LinearSVC
from sklearn import metrics

#train SVM model
svm_model = LinearSVC(C=10, dual=False)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Linear SVM accuracy = {:0.1f}%'.format(100*accuracy))

#since linear SVM accuracy is 86.5%, data is NOT linearly separable

#SVM using slack varible C
#finding best C
from sklearn import model_selection

# Define the different SVM models to use
svm_1 = LinearSVC(C = 10, dual = False)
svm_2 = LinearSVC(C = 1, dual = False)
svm_3 = LinearSVC(C = 1e-3, dual = False)
svm_4 = LinearSVC(C = 1e-7, dual = False)

split = model_selection.KFold(5)
# Get the CV scores.
cv_1 = model_selection.cross_val_score(svm_1, X_train, y_train, cv = split)
cv_2 = model_selection.cross_val_score(svm_2, X_train, y_train, cv = split)
cv_3 = model_selection.cross_val_score(svm_3, X_train, y_train, cv = split)
cv_4 = model_selection.cross_val_score(svm_4, X_train, y_train, cv = split)

# Print the average scores.
print('C = 10    CV average score = {:0.1f}%'.format(np.mean(cv_1) * 100))
print('C = 1     CV average score = {:0.1f}%'.format(np.mean(cv_2) * 100))
print('C = 1e-3  CV average score = {:0.1f}%'.format(np.mean(cv_3) * 100))
print('C = 1e-7  CV average score = {:0.1f}%'.format(np.mean(cv_4) * 100))

#so can use C=10,1,1e-3

svm_model2 = LinearSVC(C=10, dual=False)
svm_model2.fit(X_train, y_train)
svm_model2_y_pred = svm_model2.predict(X_test)
accuracy = metrics.accuracy_score(y_test, svm_model2_y_pred)
print('SVM Accuracy = {:0.1f}%'.format(100*accuracy))

#compute AUC
auc_svm = roc_auc_score(y_test, svm_model2_y_pred)
#print(auc_svm)

#get feature importance using permutation importance
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(svm_model2, X_test_scaled, y_test, scoring='roc_auc', random_state=42)
best_feature_idx_svm = np.argmax(perm_importance.importances_mean)
best_feature_svm = X.columns[best_feature_idx_svm]
#print(best_feature_svm)

#display results
svm_results_df = pd.DataFrame({
    "Model": ["Support Vector Machine"],
    "AUC Score": [auc_svm],
    "Best Predictor": [best_feature_svm]
    })

print(svm_results_df)

#compute ROC curve
fpr, tpr, _ = roc_curve(y_test, svm_model2_y_pred)

# plot  ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'SVM (AUC = {auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM')
plt.legend(loc='lower right')
plt.grid()
plt.show()
#%%

#Q3: Decision Tree

from sklearn import tree

#train decision tree model
clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)
print(np.sum(y_preds == y_test)/len(y_preds))

#compute AUC
auc_decision_tree = roc_auc_score(y_test, y_preds)

#get best feature
feature_importances = clf.feature_importances_
best_feature_idx_tree = np.argmax(feature_importances)
best_feature_tree = X.columns[best_feature_idx_tree]

#display results
svm_results_df = pd.DataFrame({
    "Model": ["Decision Tree"],
    "AUC Score": [auc_decision_tree],
    "Best Predictor": [best_feature_tree]
    })

print(svm_results_df)

#compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_preds)

#plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Decision Tree (AUC = {auc_decision_tree:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend(loc='lower right')
plt.grid()
plt.show()
#%%

#Q4: Random Forest

from sklearn.ensemble import RandomForestClassifier

#train random forest model
rf = RandomForestClassifier(n_estimators=100, max_samples=0.5, max_features=0.5, bootstrap=True, criterion='gini')
rf = rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(np.sum(y_pred == y_test)/len(y_pred))

#compute AUC
auc_random_forest = roc_auc_score(y_test, y_pred)

#get best feature
feature_importances_rf = rf.feature_importances_
best_feature_idx_rf = np.argmax(feature_importances_rf)
best_feature_rf = X.columns[best_feature_idx_rf]

#display results
rf_results_df = pd.DataFrame({
    "Model": ["Random Forest"],
    "AUC Score": [auc_random_forest],
    "Best Predictor": [best_feature_rf]
    })

print(rf_results_df)

#compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)

#plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Random Forest (AUC = {auc_random_forest:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#%%

#Q5: adaBoost

#train adaBoost model
ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=2000, learning_rate=1)
ada.fit(X_train, y_train)
pred_ada = ada.predict(X_test)
print(np.sum(pred_ada == y_test)/len(pred_ada))

#compute AUC
auc_adaBoost = roc_auc_score(y_test, pred_ada)

#get best feature
feature_importances_ada = ada.feature_importances_
best_feature_idx_ada = np.argmax(feature_importances_ada)
best_feature_ada = X.columns[best_feature_idx_ada]

#display results
ada_results_df = pd.DataFrame({
    "Model": ["AdaBoost"],
    "AUC Score": [auc_adaBoost],
    "Best Predictor": [best_feature_ada]
    })

print(ada_results_df)

#compute ROC curve
fpr, tpr, _ = roc_curve(y_test, pred_ada)

#plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AdaBoost (AUC = {auc_adaBoost:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for AdaBoost')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#%%

#Extra Credit A

model_comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "Decision Tree", "Random Forest", "AdaBoost"],
    "AUC Score": [auc_log_reg, auc_svm, auc_decision_tree, auc_random_forest, auc_adaBoost]})

print(model_comparison_df)


