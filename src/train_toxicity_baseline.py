import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


train_file = "data/processed/toxicity_train.csv"
validation_file = "data/processed/toxicity_validation.csv"
test_file = "data/processed/toxicity_test.csv"


train = pd.read_csv(train_file)
validation = pd.read_csv(validation_file)
test = pd.read_csv(test_file)


X_train = train["text_clean"]
y_train = train["toxicity_label"]

X_validation = validation["text_clean"]
y_validation = validation["toxicity_label"]

X_test = test["text_clean"]
y_test = test["toxicity_label"]


print("=== TF-IDF ===")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_validation_tfidf = vectorizer.transform(X_validation)
X_test_tfidf = vectorizer.transform(X_test)

print("Train shape     :", X_train_tfidf.shape)
print("Validation shape:", X_validation_tfidf.shape)
print("Test shape      :", X_test_tfidf.shape)


print("\n=== TRAIN LOGISTIC REGRESSION ===")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


print("\n=== VALIDATION ===")

validation_prediction = model.predict(X_validation_tfidf)

print("Accuracy :", accuracy_score(y_validation, validation_prediction))
print("Precision:", precision_score(y_validation, validation_prediction))
print("Recall   :", recall_score(y_validation, validation_prediction))
print("F1 Score :", f1_score(y_validation, validation_prediction))

print("\nClassification Report:")
print(classification_report(y_validation, validation_prediction))

print("Confusion Matrix:")
print(confusion_matrix(y_validation, validation_prediction))


print("\n=== TEST ===")

test_prediction = model.predict(X_test_tfidf)

print("Accuracy :", accuracy_score(y_test, test_prediction))
print("Precision:", precision_score(y_test, test_prediction))
print("Recall   :", recall_score(y_test, test_prediction))
print("F1 Score :", f1_score(y_test, test_prediction))

print("\nClassification Report:")
print(classification_report(y_test, test_prediction))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_prediction))