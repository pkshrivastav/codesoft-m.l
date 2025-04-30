# CUSTOMER CHURN PREDICTION
# Author: Prince Kumar
# -----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Step 1: Load Dataset
# -------------------------------

df = pd.read_csv("Churn_Modelling.csv")
print("First 5 rows:\n", df.head(), "\n")
print("Shape:", df.shape)

# Rename 'Exited' to 'Churn' for clarity
df.rename(columns={'Exited': 'Churn'}, inplace=True)

# -------------------------------
# Step 2: Exploratory Data Analysis
# -------------------------------

print("\nChurn Count:\n", df['Churn'].value_counts())
sns.countplot(x='Churn', data=df)
plt.title("Churn Count")
plt.show()

# Churn by Gender
sns.countplot(x='Gender', hue='Churn', data=df)
plt.title("Churn by Gender")
plt.show()

# Churn by Geography
sns.countplot(x='Geography', hue='Churn', data=df)
plt.title("Churn by Geography")
plt.show()

# -------------------------------
# Step 3: Preprocessing
# -------------------------------

# Drop unnecessary columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# Feature-target split
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -------------------------------
# Step 4: Model Training & Evaluation
# -------------------------------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n[Logistic Regression]")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n[Random Forest]")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print("\n[Gradient Boosting]")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print(confusion_matrix(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

# -------------------------------
# Step 5: Feature Importance (Gradient Boosting)
# -------------------------------

importances = gb.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importance (Gradient Boosting)")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 6: Predict a new customer
# -------------------------------

new_customer = [[600, 1, 40, 3, 60000, 2, 1, 1, 50000, 0, 0]]  # sample scaled data
new_customer = sc.transform(new_customer)
print("\nNew Customer Churn Prediction (1 = Churn):", gb.predict(new_customer))

# -------------------------------
# DONE!
# -------------------------------
