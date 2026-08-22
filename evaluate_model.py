import joblib
import pandas as pd
import matplotlib.pyplot as plt

from utils.preprocessing import clean_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================
# PREPROCESS TEXT
# ==========================================

print("Cleaning text...")

data["title"] = data["title"].apply(clean_text)
data["text"] = data["text"].apply(clean_text)

data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["label"]

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# LOAD SAVED MODEL AND VECTORIZER
# ==========================================

print("Loading saved model...")

model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Convert test data into TF-IDF features
X_test = vectorizer.transform(X_test)

# ==========================================
# PREDICTION
# ==========================================

print("Generating predictions...")

predictions = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\n=========================================")
print("MODEL EVALUATION")
print("=========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

print("\nClassification Report:")
report = classification_report(
    y_test,
    predictions,
    target_names=["Fake News", "Real News"]
)

print(report)

# ==========================================
# SAVE CLASSIFICATION REPORT
# ==========================================

with open("evaluation_report.txt", "w") as file:
    file.write("FAKE NEWS DETECTION - MODEL EVALUATION\n")
    file.write("=" * 45 + "\n\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write("Confusion Matrix:\n")
    file.write(str(cm))
    file.write("\n\nClassification Report:\n")
    file.write(report)

# ==========================================
# CONFUSION MATRIX IMAGE
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake News", "Real News"]
)

display.plot()
plt.title("Fake News Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig("screenshots/confusion_matrix.png")

print("\nEvaluation completed successfully!")
print("Saved: evaluation_report.txt")
print("Saved: screenshots/confusion_matrix.png")