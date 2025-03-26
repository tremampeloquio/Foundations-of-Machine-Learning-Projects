# Machine Learning Projects 📊🤖  
A collection of applied machine learning projects from my coursework at NYU, focused on predictive modeling, regression analysis, and classification. These projects explore different modeling techniques and apply them to real-world datasets, demonstrating both technical implementation and interpretability of results.

---

## 📁 Projects

### 1. Housing Value Regression Model 🏘️  
**Objective:** Predict median housing value in California based on socioeconomic and structural features.  
**Techniques Used:**  
- Data normalization & correlation analysis  
- Simple and multiple linear regression  
- Model evaluation using R² and residual diagnostics  

**Key Insight:**  
Median Income was the strongest single predictor of housing value (R² = 0.473), while the full multiple regression model explained 60.1% of the variance (R² = 0.601). Normalizing variables like Rooms and Bedrooms by population improved correlation.  

📄 [Full Report →]([./Housing_Analysis_Report.pdf](https://github.com/tremampeloquio/Foundations-of-Machine-Learning-Projects/blob/main/Housing%20Analysis/Trem%20Ampeloquio%2C%20Housing_Analysis_Report.pdf))

---

### 2. Tech Salaries Regression + Bias Analysis 💼💰  
**Objective:** Predict total annual compensation in the tech industry and investigate gender pay gap claims.  
**Techniques Used:**  
- Multiple linear regression, Ridge, Lasso  
- Logistic regression for classification  
- Bias & fairness analysis, Shapiro-Wilk normality testing  
- Violin plots for job title-based compensation insights  

**Key Insight:**  
Years of relevant experience was the strongest salary predictor (R² = 0.1766), with the full model reaching R² = 0.2871. Lasso shrunk few coefficients, and logistic regression failed to predict gender from compensation, indicating potential data imbalance.

📄 [Full Report →]([./Tech_Salaries_Report%202.pdf](https://github.com/tremampeloquio/Foundations-of-Machine-Learning-Projects/blob/main/Housing%20Analysis/Trem%20Ampeloquio%2C%20Housing_Analysis_Report.pdf))

---

### 3. Diabetes Prediction Models 🧬🩺  
**Objective:** Compare classification models to predict likelihood of diabetes using health-related variables.  
**Models Used:**  
- Logistic Regression (AUC = 0.83 -> Best Model)  
- Support Vector Machine  
- Decision Tree  
- Random Forest  
- AdaBoost  

**Key Insight:**  
Logistic Regression outperformed other models in predictive power (AUC = 0.83), with *General Health* being the most influential predictor. Models like AdaBoost and SVM performed only slightly better than chance (AUC ~ 0.5–0.6), suggesting the data benefits from simpler, interpretable models.

📄 [Full Report →]([./Diabetes_Prediction_Report.pdf](https://github.com/tremampeloquio/Foundations-of-Machine-Learning-Projects/blob/main/Diabetes/Trem%20Ampeloquio%2C%20Diabetes%20Prediction%20Report.pdf))

---

## 🧰 Tools & Languages  
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook, Spyder 

---

## 📬 Contact  
Made by [Trem Ampeloquio](https://www.linkedin.com/in/trem-ampeloquio-b2a84a2a5)  
📧 tremampeloquio@gmail.com  

