import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data
train_data = pd.read_csv('synthetic_spam_dataset.csv')  
print("Training data loaded.")

# Check for missing values
print("Missing values in each column:")
print(train_data.isnull().sum())

# Separate features and target variable
X = train_data.drop('target', axis=1)
y = train_data['target']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# --- Training Naive Bayes Model ---
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_val)

# Evaluate Naive Bayes Model
nb_accuracy = accuracy_score(y_val, nb_predictions)
nb_precision = precision_score(y_val, nb_predictions)
nb_recall = recall_score(y_val, nb_predictions)
nb_f1 = f1_score(y_val, nb_predictions)

print("--- Training Naive Bayes Model ---")
print(f"Naive Bayes - Accuracy: {nb_accuracy}")
print(f"Naive Bayes - Precision: {nb_precision}")
print(f"Naive Bayes - Recall: {nb_recall}")
print(f"Naive Bayes - F1-score: {nb_f1}")

# --- Training SVM Model ---
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_val)

# Evaluate SVM Model
svm_accuracy = accuracy_score(y_val, svm_predictions)
svm_precision = precision_score(y_val, svm_predictions)
svm_recall = recall_score(y_val, svm_predictions)
svm_f1 = f1_score(y_val, svm_predictions)

print("--- Training SVM Model ---")
print(f"SVM - Accuracy: {svm_accuracy}")
print(f"SVM - Precision: {svm_precision}")
print(f"SVM - Recall: {svm_recall}")
print(f"SVM - F1-score: {svm_f1}")

# --- Training Logistic Regression Model ---
log_reg_model = LogisticRegression(max_iter=2000)
log_reg_model.fit(X_train, y_train)
log_reg_predictions = log_reg_model.predict(X_val)

# Evaluate Logistic Regression Model
log_reg_accuracy = accuracy_score(y_val, log_reg_predictions)
log_reg_precision = precision_score(y_val, log_reg_predictions)
log_reg_recall = recall_score(y_val, log_reg_predictions)
log_reg_f1 = f1_score(y_val, log_reg_predictions)

print("--- Training Logistic Regression Model ---")
print(f"Logistic Regression - Accuracy: {log_reg_accuracy}")
print(f"Logistic Regression - Precision: {log_reg_precision}")
print(f"Logistic Regression - Recall: {log_reg_recall}")
print(f"Logistic Regression - F1-score: {log_reg_f1}")

# --- Load the synthetic test data ---
test_data = pd.read_csv('synthetic_test_data.csv') 
print("Test data loaded.")

# Separate features and target variable for the test set
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# Evaluate on Naive Bayes Model
nb_test_predictions = nb_model.predict(X_test)
nb_test_accuracy = accuracy_score(y_test, nb_test_predictions)
nb_test_precision = precision_score(y_test, nb_test_predictions)
nb_test_recall = recall_score(y_test, nb_test_predictions)
nb_test_f1 = f1_score(y_test, nb_test_predictions)

print("--- Evaluating Naive Bayes on Test Data ---")
print(f"Naive Bayes - Test Accuracy: {nb_test_accuracy}")
print(f"Naive Bayes - Test Precision: {nb_test_precision}")
print(f"Naive Bayes - Test Recall: {nb_test_recall}")
print(f"Naive Bayes - Test F1-score: {nb_test_f1}")

# Repeat similar evaluation for SVM and Logistic Regression
svm_test_predictions = svm_model.predict(X_test)
svm_test_accuracy = accuracy_score(y_test, svm_test_predictions)
svm_test_precision = precision_score(y_test, svm_test_predictions)
svm_test_recall = recall_score(y_test, svm_test_predictions)
svm_test_f1 = f1_score(y_test, svm_test_predictions)

print("--- Evaluating SVM on Test Data ---")
print(f"SVM - Test Accuracy: {svm_test_accuracy}")
print(f"SVM - Test Precision: {svm_test_precision}")
print(f"SVM - Test Recall: {svm_test_recall}")
print(f"SVM - Test F1-score: {svm_test_f1}")

log_reg_test_predictions = log_reg_model.predict(X_test)
log_reg_test_accuracy = accuracy_score(y_test, log_reg_test_predictions)
log_reg_test_precision = precision_score(y_test, log_reg_test_predictions)
log_reg_test_recall = recall_score(y_test, log_reg_test_predictions)
log_reg_test_f1 = f1_score(y_test, log_reg_test_predictions)

print("--- Evaluating Logistic Regression on Test Data ---")
print(f"Logistic Regression - Test Accuracy: {log_reg_test_accuracy}")
print(f"Logistic Regression - Test Precision: {log_reg_test_precision}")
print(f"Logistic Regression - Test Recall: {log_reg_test_recall}")
print(f"Logistic Regression - Test F1-score: {log_reg_test_f1}")
