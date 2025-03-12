# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('spambase.data', header=None)

# Step 1: Separate Features and Labels
X = data.iloc[:, :-1]  # All columns except the last one are features
y = data.iloc[:, -1]   # The last column is the label

# Step 2: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preview the data shapes to confirm
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Naive Bayes Model
print("\n--- Training Naive Bayes Model ---")
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nNaive Bayes Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Support Vector Machine (SVM) Model
print("\n--- Training SVM Model ---")
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
print("SVM model training complete.")

# Predict using the SVM model
y_pred_svm = svm_model.predict(X_test)
print("SVM model prediction complete.")

# Evaluate the SVM model
print("\nSVM Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm))


# Logistic Regression Model
print("\n--- Training Logistic Regression Model ---")
logreg_model = LogisticRegression(max_iter=3000)
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)
print("\nLogistic Regression Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Precision:", precision_score(y_test, y_pred_logreg))
print("Recall:", recall_score(y_test, y_pred_logreg))
print("F1 Score:", f1_score(y_test, y_pred_logreg))


