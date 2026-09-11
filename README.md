# Telco Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer churn is an important business problem for telecom companies. Identifying customers who are likely to leave the service can help companies take preventive actions and improve customer retention.

This project uses Machine Learning to predict whether a telecom customer will **churn or not churn** based on customer information, services, contract details, payment method, tenure, and billing information.

The project was implemented using **Python and Scikit-learn** in Jupyter Notebook.

---

## 🎯 Objective

The main objective of this project is to:

* Analyze customer data.
* Understand customer churn patterns.
* Clean and preprocess the dataset.
* Perform Exploratory Data Analysis (EDA).
* Convert categorical variables into numerical form.
* Split the dataset into training and testing sets.
* Perform feature scaling.
* Build a Logistic Regression classification model.
* Evaluate the model using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
* Apply 5-Fold Cross-Validation.
* Perform Hyperparameter Tuning using GridSearchCV.
* Perform Hyperparameter Tuning using RandomizedSearchCV.
* Compare model performance.

---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

### Dataset Information

* **Rows:** 7,043
* **Columns:** 21
* **Target Variable:** `Churn`

The dataset contains customer-related information such as:

* Customer ID
* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges
* Churn

The dataset contains **5,174 customers who did not churn** and **1,869 customers who churned**.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 🔄 Project Workflow

### 1. Load Dataset

The dataset was loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

### 2. Data Understanding

The dataset was examined using:

* `head()`
* `tail()`
* `info()`
* `shape`
* `value_counts()`

The dataset contains 7,043 records and 21 columns.

### 3. Data Cleaning

The following preprocessing steps were performed:

* Checked duplicate records.
* Checked missing values.
* Converted `TotalCharges` from object type to numerical type.

```python
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)
```

### 4. Exploratory Data Analysis

EDA was performed to understand the distribution of customer churn and relationships between customer attributes and churn.

Visualization libraries used:

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

### 5. Categorical Encoding

Categorical variables were converted into numerical form so they could be used by the machine learning model.

### 6. Train-Test Split

The dataset was divided into training and testing data.

The test dataset contains approximately **20% of the records**.

### 7. Feature Scaling

Feature scaling was performed before training the Logistic Regression model.

### 8. Logistic Regression

Logistic Regression was used as the primary classification algorithm for predicting customer churn.

### 9. Model Evaluation

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* Classification Report

### 10. 5-Fold Cross-Validation

5-Fold Cross-Validation was used to evaluate the model more reliably across multiple subsets of the training data.

### 11. GridSearchCV

GridSearchCV was used to find the best Logistic Regression hyperparameters.

The best parameters found were:

```text
C = 1.0
penalty = l2
solver = lbfgs
max_iter = 100
```

The best cross-validation score was approximately:

```text
0.8048
```

### 12. RandomizedSearchCV

RandomizedSearchCV was also used for hyperparameter tuning and model evaluation.

### 13. Model Comparison

The performance of:

* Logistic Regression
* GridSearchCV
* RandomizedSearchCV

was compared using test accuracy.

---

## 📈 Model Performance

The tuned Logistic Regression model achieved approximately:

**Test Accuracy: 80.70%**

### Classification Performance

| Metric    | No Churn | Churn |
| --------- | -------: | ----: |
| Precision |     0.85 |  0.66 |
| Recall    |     0.89 |  0.57 |
| F1-score  |     0.87 |  0.61 |

The confusion matrix was:

```text
[[925 110]
 [162 212]]
```

The model therefore identifies customers who churn as well as customers who stay, with the churn class being the more challenging class to predict.

---

## 💡 Business Understanding

A telecom company can use this type of model to identify customers who have a higher probability of leaving.

Potential business applications include:

* Identifying high-risk customers.
* Improving customer retention.
* Offering targeted discounts.
* Providing personalized plans.
* Improving customer support.
* Designing retention campaigns.

---

## 📁 Project Structure

```text
Telco-Customer-Churn/
│
├── Telco_Customer_Churn.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
└── images/
    └── visualizations/
```

---

## 🚀 How to Run the Project

### 1. Clone or download the project

Download the project files to your computer.

### 2. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Open Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
Telco_Customer_Churn.ipynb
```

### 5. Run the cells

Run the notebook cells sequentially to reproduce the analysis and machine learning results.

---

## 📌 Key Skills Demonstrated

* Data Cleaning
* Data Preprocessing
* Exploratory Data Analysis
* Data Visualization
* Categorical Encoding
* Feature Scaling
* Train-Test Split
* Logistic Regression
* Confusion Matrix
* Classification Report
* Accuracy, Precision, Recall and F1-score
* Cross-Validation
* GridSearchCV
* RandomizedSearchCV
* Hyperparameter Tuning
* Model Comparison

---

## 🏁 Conclusion

The Telco Customer Churn dataset was successfully analyzed and used to build a machine learning classification model for predicting customer churn.

Logistic Regression was used as the primary model, and hyperparameter optimization was performed using GridSearchCV and RandomizedSearchCV. The tuned model achieved approximately **80.7% test accuracy**.

This project demonstrates an end-to-end Machine Learning workflow, from data understanding and preprocessing to model training, evaluation, and hyperparameter tuning.

---

## 👨‍💻 Author

**Rohit Tayde**

**Project:** Telco Customer Churn Prediction Using Machine Learning
