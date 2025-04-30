# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Load the dataset
# data = pd.read_csv("spam.csv", encoding='ISO-8859-1', on_bad_lines='skip')

# # Rename columns
# data.columns = ['Category', 'Message', 'Column3', 'Column4', 'Column5']

# # Remove unnecessary columns
# data = data[['Category', 'Message']]

# # Clean the 'Category' column (ham = 0, spam = 1)
# data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# # Plot: Distribution of Ham vs Spam messages
# plt.figure(figsize=(8, 6))
# sns.countplot(x='Category', data=data, palette='Set2')
# plt.title('Distribution of Ham vs Spam Messages')
# plt.xticks([0, 1], ['Ham', 'Spam'])
# plt.ylabel('Number of Messages')
# plt.show()

# # Split the data into features (X) and labels (y)
# X = data['Message']
# y = data['Category']

# # Convert text messages to feature vectors using TF-IDF
# tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
# X_tfidf = tfidf.fit_transform(X)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# # Initialize the Naive Bayes model
# model = MultinomialNB()

# # Train the model on the training data
# model.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Plot: Confusion Matrix (Heatmap)
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # Plot: Top 20 Most Frequent Words in the Dataset
# word_freq = np.array(tfidf.get_feature_names_out())
# sorted_idx = np.argsort(np.asarray(tfidf.idf_))[:20]
# top_words = word_freq[sorted_idx]
# top_word_frequencies = np.sort(np.asarray(tfidf.idf_))[:20]

# plt.figure(figsize=(10, 6))
# sns.barplot(x=top_word_frequencies, y=top_words, palette='viridis')
# plt.title('Top 20 Most Frequent Words (TF-IDF)')
# plt.xlabel('TF-IDF Score')
# plt.ylabel('Words')
# plt.show()

# # Plot: Precision, Recall, F1-Score for the Classifier
# report = classification_report(y_test, y_pred, output_dict=True)
# metrics = ['precision', 'recall', 'f1-score']
# labels = ['ham', 'spam']
# for metric in metrics:
#     plt.figure(figsize=(8, 6))
#     scores = [report[label][metric] for label in labels]
#     sns.barplot(x=labels, y=scores, palette='Set1')
#     plt.title(f'{metric.capitalize()} for Ham vs Spam')
#     plt.ylabel(metric.capitalize())
#     plt.show()

# # Example: Predict if a new message is spam or ham
# new_message = ["Congratulations! You've won a free ticket to the concert. Call now!"]
# new_message_tfidf = tfidf.transform(new_message)
# prediction = model.predict(new_message_tfidf)
# print("Prediction for new message:", "Spam" if prediction[0] == 1 else "Ham")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Load dataset
data = pd.read_csv("spam.csv", encoding='ISO-8859-1', on_bad_lines='skip')

# Clean the data and rename columns
data.columns = ['Category', 'Message', 'Column3', 'Column4', 'Column5']
data = data[['Category', 'Message']]
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(8, 6))
sns.countplot(x='Category', data=data, palette='Set2')
plt.title('Distribution of Ham vs Spam Messages')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.ylabel('Number of Messages')
plt.show()

# Prepare features and labels
X = data['Message']
y = data['Category']

# TF-IDF Vectorization with bigrams and unigrams
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Naive Bayes using GridSearchCV
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 3.0]}
nb_model = MultinomialNB()
grid_search_nb = GridSearchCV(nb_model, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search_nb.fit(X_train, y_train)

# Best model from grid search
best_nb_model = grid_search_nb.best_estimator_
print(f"Best Naive Bayes alpha: {grid_search_nb.best_params_['alpha']}")

# Train the model
best_nb_model.fit(X_train, y_train)

# Make predictions
y_pred = best_nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix (Heatmap)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_nb_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label='Precision-Recall curve')
plt.fill_between(recall, precision, color='blue', alpha=0.2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_nb_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Compare different models (Naive Bayes, Logistic Regression, SVM)
models = {
    'Naive Bayes': best_nb_model,
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(kernel='linear', probability=True)
}

for name, model in models.items():
    # Cross-validation for each model
    cv_scores = cross_val_score(model, X_tfidf, y, cv=5)
    print(f"{name} Cross-validation Accuracy: {np.mean(cv_scores):.4f}")

# Feature importance visualization
features = np.array(tfidf.get_feature_names_out())
top_n = 20
coefficients = best_nb_model.feature_log_prob_[1]  # spam class
top_n_idx = np.argsort(coefficients)[-top_n:]
top_n_features = features[top_n_idx]
top_n_coefficients = coefficients[top_n_idx]
