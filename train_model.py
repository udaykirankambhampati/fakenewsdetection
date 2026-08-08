import joblib
import pandas as pd
from utils.preprocessing import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# LOAD DATASET
# ==========================================
print("Loading datasets...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Merge datasets
data = pd.concat([fake, true], ignore_index=True)

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Datasets loaded successfully!")

# ==========================================
# PREPROCESS TEXT
# ==========================================

print("Cleaning text...")

data["title"] = data["title"].apply(clean_text)
data["text"] = data["text"].apply(clean_text)

# Combine title and text
data["content"] = data["title"] + " " + data["text"]

print("Text preprocessing completed!")

# ==========================================
# FEATURES & LABELS
# ==========================================

X = data["content"]
y = data["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

print("Performing TF-IDF Vectorization...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("TF-IDF completed!")

print("\nTraining Data Shape :", X_train.shape)
print("Testing Data Shape  :", X_test.shape)

# ==========================================
# LOGISTIC REGRESSION
# ==========================================

print("\n=========================================")
print("LOGISTIC REGRESSION")
print("=========================================")

lr_model = LogisticRegression(max_iter=1000)

print("Training Logistic Regression Model...")
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\nAccuracy:", round(lr_accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

print("\nClassification Report:")
print(classification_report(y_test, lr_predictions))

# ==========================================
# PASSIVE AGGRESSIVE CLASSIFIER
# ==========================================

print("\n=========================================")
print("PASSIVE AGGRESSIVE CLASSIFIER")
print("=========================================")

pac_model = PassiveAggressiveClassifier(
    max_iter=1000,
    random_state=42
)

print("Training Passive Aggressive Model...")
pac_model.fit(X_train, y_train)

pac_predictions = pac_model.predict(X_test)

pac_accuracy = accuracy_score(y_test, pac_predictions)

print("\nAccuracy:", round(pac_accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pac_predictions))

print("\nClassification Report:")
print(classification_report(y_test, pac_predictions))

# ==========================================
# MODEL COMPARISON
# ==========================================

print("\n=========================================")
print("MODEL COMPARISON")
print("=========================================")

print(f"Logistic Regression Accuracy      : {lr_accuracy * 100:.2f}%")
print(f"Passive Aggressive Accuracy       : {pac_accuracy * 100:.2f}%")

if lr_accuracy > pac_accuracy:
    print("\nBest Model : Logistic Regression")
elif pac_accuracy > lr_accuracy:
    print("\nBest Model : Passive Aggressive Classifier")
else:
    print("\nBoth models performed equally well.")

# ==========================================
# SAVE BEST MODEL
# ==========================================

if lr_accuracy >= pac_accuracy:
    best_model = lr_model
    model_name = "Logistic Regression"
else:
    best_model = pac_model
    model_name = "Passive Aggressive Classifier"

# Save model
joblib.dump(best_model, "model/fake_news_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\n=========================================")
print("MODEL SAVED SUCCESSFULLY")
print("=========================================")

print(f"Best Model : {model_name}")
print("Saved Model : model/fake_news_model.pkl")
print("Saved Vectorizer : model/tfidf_vectorizer.pkl")