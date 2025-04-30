import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("movies.csv", encoding="ISO-8859-1")

# Display basic information
print(df.head())
print(df.info())

# Select relevant features
features = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3", "Year", "Duration", "Votes"]
target = "Rating"

# Drop rows where target is missing
df = df.dropna(subset=[target])

# Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
df[features] = imputer.fit_transform(df[features])

# Encode categorical variables
encoder = LabelEncoder()
for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
    df[col] = encoder.fit_transform(df[col])

# Convert 'Year' and 'Votes' to numeric
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# Ensure 'Duration' is numeric (removing 'min' and converting to float)
df["Duration"] = df["Duration"].str.extract("(\d+)").astype(float)

# Fill any remaining missing values with the median
df.fillna(df.median(numeric_only=True), inplace=True)

# Split dataset
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for NaN values before scaling
print("Missing values in X_train:", np.isnan(X_train).sum().sum())
print("Missing values in X_test:", np.isnan(X_test).sum().sum())

# Replace NaN with zero if necessary
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
lr_model = LinearRegression()

rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Evaluate models
def evaluate(y_true, y_pred, model_name):
    print(f"\n🔹 Model: {model_name}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.3f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
    print(f"R2 Score: {r2_score(y_true, y_pred):.3f}")

evaluate(y_test, rf_pred, "Random Forest")
evaluate(y_test, lr_pred, "Linear Regression")

# --- DIAGRAMS ---

# 1. Distribution of Ratings
plt.figure(figsize=(8, 5))
sns.histplot(df["Rating"], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# 2. Genre Popularity
plt.figure(figsize=(10, 5))
sns.countplot(y=df["Genre"], order=df["Genre"].value_counts().index, palette="coolwarm")
plt.title("Number of Movies per Genre")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()

# 3. Top 10 Directors with Most Movies
plt.figure(figsize=(10, 5))
top_directors = df["Director"].value_counts().nlargest(10)
sns.barplot(x=top_directors.values, y=top_directors.index, palette="viridis")
plt.title("Top 10 Directors with Most Movies")
plt.xlabel("Number of Movies")
plt.ylabel("Director")
plt.show()

# 4. Feature Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Feature Importance (Random Forest)
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="magma")
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature Name")
plt.show()

# 6. Residual Plot (Random Forest)
plt.figure(figsize=(8, 5))
sns.residplot(x=y_test, y=rf_pred, lowess=True, color="blue")
plt.title("Residual Plot (Random Forest)")
plt.xlabel("Actual Ratings")
plt.ylabel("Residuals")
plt.show()

# 7. Residual Plot (Linear Regression)
plt.figure(figsize=(8, 5))
sns.residplot(x=y_test, y=lr_pred, lowess=True, color="red")
plt.title("Residual Plot (Linear Regression)")
plt.xlabel("Actual Ratings")
plt.ylabel("Residuals")
plt.show()

# 8. Actual vs Predicted Ratings (Both Models)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=rf_pred, alpha=0.6, label="Random Forest", color="blue")
sns.scatterplot(x=y_test, y=lr_pred, alpha=0.6, label="Linear Regression", color="red")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='black')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings (Both Models)")
plt.legend()
plt.show()

# 9. Boxplot of Ratings by Genre
plt.figure(figsize=(12, 6))
sns.boxplot(x=df["Genre"], y=df["Rating"], palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Boxplot of Ratings by Genre")
plt.xlabel("Genre")
plt.ylabel("Rating")
plt.show()

# 10. Scatter Plot of Votes vs. Ratings
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Votes"], y=df["Rating"], alpha=0.6, color="purple")
plt.title("Votes vs. Ratings")
plt.xlabel("Votes")
plt.ylabel("Rating")
plt.xscale("log")  # Log scale to handle large vote numbers
plt.show()
