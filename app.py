from flask import Flask, render_template, request
import joblib
import math
from utils.preprocessing import clean_text

app = Flask(__name__)

# ==========================================
# LOAD BEST MODEL
# ==========================================

model = joblib.load("model/fake_news_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    # Keep user input after prediction
    title = ""
    content = ""

    if request.method == "POST":

        # Get user input
        title = request.form["title"]
        content = request.form["content"]

        # Clean text
        clean_title = clean_text(title)
        clean_content = clean_text(content)

        # Combine title and content
        text = clean_title + " " + clean_content

        # Convert text into TF-IDF features
        features = vectorizer.transform([text])

        # ==========================================
        # MAKE PREDICTION
        # ==========================================

        result = model.predict(features)[0]

        # ==========================================
        # CONFIDENCE
        # Passive Aggressive does not support
        # predict_proba(), so use decision_function()
        # ==========================================

        decision_score = model.decision_function(features)[0]

        # Convert decision score into a confidence-like value
        probability = 1 / (1 + math.exp(-abs(decision_score)))

        confidence = round(probability * 100, 2)

        # ==========================================
        # CONVERT RESULT
        # ==========================================

        if result == 0:
            prediction = "FAKE NEWS"
        else:
            prediction = "REAL NEWS"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        title=title,
        content=content
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)