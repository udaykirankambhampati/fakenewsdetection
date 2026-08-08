import joblib
from utils.preprocessing import clean_text

# Load saved model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

def predict_news(title, text):
    # Clean input
    title = clean_text(title)
    text = clean_text(text)

    # Combine title and text
    content = title + " " + text

    # Vectorize
    vector = vectorizer.transform([content])

    # Predict
    prediction = model.predict(vector)[0]

    # Confidence Score
    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(vector)[0]) * 100
    elif hasattr(model, "decision_function"):
        score = abs(model.decision_function(vector)[0])
        confidence = min(99.99, 50 + score)
    else:
        confidence = None

    if prediction == 1:
        result = "REAL NEWS"
    else:
        result = "FAKE NEWS"

    return result, confidence


# Test
if __name__ == "__main__":

    title = input("Enter News Title:\n")
    text = input("\nEnter News Content:\n")

    result, confidence = predict_news(title, text)

    print("\nPrediction :", result)

    if confidence is not None:
        print(f"Confidence : {confidence:.2f}%")