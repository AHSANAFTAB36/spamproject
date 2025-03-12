#  Spam Detection Project

##  Overview
This project is designed to classify messages as *spam or legitimate* using machine learning models. The models implemented include **Naive Bayes, Support Vector Machine (SVM), and Logistic Regression**. The dataset used for training includes the **UCI Spambase dataset** and a **synthetic dataset**.

##  Dataset
- UCI Spambase Dataset (spambase.data) - Preprocessed dataset with numerical features.
- Synthetic Spam Dataset (synthetic_spam_dataset.csv) - Used for additional testing.

##  Preprocessing
- The dataset is already structured with numerical features.
- No text preprocessing (stopwords removal, stemming, etc.) was required.
- The data was split into *training (80%)* and **testing (20%)**.

##  Models Used
The following models were implemented and evaluated:
1. *Naive Bayes* - Efficient for high-dimensional data.
2. *Support Vector Machine (SVM)* - Used with RBF kernel for improved performance.
3. *Logistic Regression* - Performed best among the models.

##  Performance Summary
| Model  | Accuracy | Precision | Recall | F1 Score |
|--------|----------|------------|--------|----------|
| Naive Bayes  | 78.6% | 76.4% | 71.5% | 73.9% |
| SVM  | 66.2% | 66.1% | 41.5% | 51.0% |
| Logistic Regression | *92.3%* | *93.2%* | *88.2%* | *90.6%* |

##  How to Run the Project
### *1. Install Dependencies*
Make sure you have Python installed, then install the required libraries:
```sh
pip install pandas numpy scikit-learn
```

### 2. Run the Script
To execute the main spam detection script, run:
```sh
python spam_detection.py
```
For the second model:
```sh
python newspam_detection.py
```

##  Evaluating a New Message
To classify new messages, use the trained **Logistic Regression model**:
```python
y_pred_new = logreg_model.predict(new_data)
print(y_pred_new)  # Output: 1 (spam) or 0 (legitimate)
```

##  Project Repository
The complete code and dataset for this project can be found on GitHub:
 [https://github.com/yourusername/Spam-Detection-Project](https://github.com/yourusername/Spam-Detection-Project)

##  Future Improvements
- Implement deep learning models (LSTM, BERT) for better performance.
- Include raw text preprocessing (tokenization, TF-IDF, etc.).
- Test on progressively larger datasets.

## Contact
For any questions or clarifications, reach out at ahsan.aftab@icloud.com.

