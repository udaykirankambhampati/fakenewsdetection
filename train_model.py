import pandas as pd
from utils.preprocessing import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Merge datasets
data = pd.concat([fake, true], ignore_index=True)

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text
data["title"] = data["title"].apply(clean_text)
data["text"] = data["text"].apply(clean_text)

# Combine title and text
data["content"] = data["title"] + " " + data["text"]

# Features and labels
X = data["content"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

print("Training Logistic Regression Model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))